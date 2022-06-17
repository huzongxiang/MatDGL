# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:21:03 2022

@author: huzongxiang
adapted from tensorflow official strategy for multi-GPU trainning "Custom training with tf.distribute.Strategy"
https://tensorflow.google.cn/tutorials/distribute/custom_training
"""


import time
import warnings
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


ModulePath = Path(__file__).parent.absolute()


class Pretrainer_dist:
    def __init__(self,
        model: Model,
        atom_dim=16,
        bond_dim=64,
        num_atom=119,
        state_dim=16,
        sp_dim=230,
        units=32,
        edge_steps=1,
        transform_steps=1,
        num_attention_heads=8,
        dense_units=64,
        reg0=0.00,
        reg1=0.00,
        batch_size=32,
        spherical_harmonics=True,
        final_dim=119,
        ):
        self.model = model
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_atom = num_atom
        self.state_dim = state_dim
        self.sp_dim = sp_dim
        self.final_dim = final_dim
        self.units = units
        self.edge_steps = edge_steps
        self.transform_steps = transform_steps
        self.num_attention_heads = num_attention_heads
        self.dense_units = dense_units
        self.reg0 = reg0
        self.reg1 = reg1
        self.batch_size = batch_size
        self.spherical_harmonics = spherical_harmonics
        self.final_dim = final_dim


    def train(self, train_data, valid_data, epochs=10, lr=1e-3, workdir=None):

        # Distribute strategy
        strategy = tf.distribute.MirroredStrategy()

        batch_size_per_replica = self.batch_size

        # Global batch size
        GLOBAL_BATCH_SIZE = batch_size_per_replica * strategy.num_replicas_in_sync

        # Buffer size for data loader
        BUFFER_SIZE = batch_size_per_replica * strategy.num_replicas_in_sync * 16

        # distribute dataset
        train_dist_dataset = strategy.experimental_distribute_dataset(train_data)
        valid_dist_dataset = strategy.experimental_distribute_dataset(valid_data)

        # Create a checkpoint directory to store the checkpoints
        Path(workdir/"model").mkdir(exist_ok=True)
        Path(workdir/"model/pretrained").mkdir(exist_ok=True)
        checkpoint_dir = Path(workdir/"model/pretrained").mkdir(exist_ok=True)

        # strategy
        with strategy.scope():

            # model
            model = self.model(atom_dim=self.atom_dim,
                                bond_dim=self.bond_dim,
                                num_atom=self.num_atom,
                                state_dim=self.state_dim,
                                sp_dim=self.sp_dim,
                                units=self.units,
                                edge_steps=self.edge_steps,
                                transform_steps=self.transform_steps,
                                num_attention_heads=self.num_attention_heads,
                                dense_units=self.dense_units,
                                reg0=self.reg0,
                                reg1=self.reg1,
                                batch_size=self.batch_size,
                                spherical_harmonics=self.spherical_harmonics,
                                final_dim=self.final_dim,
                                )
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
            # checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=10, checkpoint_name='ckpt')

            # loss
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            def compute_loss(logits, labels):
                per_example_loss = loss_object(y_true=labels, y_pred=logits)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

            valid_loss = tf.keras.metrics.Mean(name='valid_loss')

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
            valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


            def train_step(dataset):
                inputs, labels = dataset

                with tf.GradientTape() as tape:
                    logits = model(inputs, training=True)

                    loss = compute_loss(logits=logits, labels=labels)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_accuracy.update_state(labels, logits)

                return loss


            def valid_step(dataset):
                inputs, labels = dataset
                predictions = model(inputs, training=False)
                t_loss = loss_object(labels, predictions)

                valid_loss.update_state(t_loss)
                valid_accuracy.update_state(labels, predictions)


            @tf.function
            def distributed_train_step(dataset_inputs):
                per_replica_losses = strategy.run(
                    train_step, args=(dataset_inputs,)
                )
                return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


            @tf.function
            def distributed_valid_step(dataset_inputs):
                return strategy.run(valid_step, args=(dataset_inputs,))


            for epoch in range(epochs):
                # TRAIN LOOP
                total_loss = 0.0
                num_batches = 0
                # dist_iterator = iter(train_dist_dataset)
                for x in train_dist_dataset:
                    total_loss += distributed_train_step(x)
                    num_batches += 1
                train_loss = total_loss / num_batches

                # TEST LOOP
                for x in valid_dist_dataset:
                    distributed_valid_step(x)

                # if epoch % 2 == 0:
                #     checkpoint_manager.save()

                template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                            "Test Accuracy: {}")
                print (template.format(epoch+1, train_loss,
                                        train_accuracy.result()*100, valid_loss.result(),
                                        valid_accuracy.result()*100))

                valid_loss.reset_states()
                train_accuracy.reset_states()
                valid_accuracy.reset_states()
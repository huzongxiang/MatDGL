# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:52:10 2022

@author: huzongxiang
"""

from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints


class GraphAttentionLayer(layers.Layer):
    """
    The GAT implementation as described in the paper
    Graph Attention Networks (Veličković et al., ICLR 2018): https://arxiv.org/abs/1710.10903
    """
    def __init__(self,
        steps=1,
        num_heads=8,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        recurrent_regularizer=None,
        bias_constraint=None,
        activation=None,
        **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.num_heads = num_heads
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        
        with tf.name_scope("self-attention"):
            # weight for updating atom_features
            self.kernel = self.add_weight(
                shape=(self.atom_dim, 
                       self.atom_dim * self.num_heads),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='node_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.atom_dim * self.num_heads,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='node_bias',
            )

            self.kernel_edge = self.add_weight(
                shape=(self.edge_dim, 
                       self.edge_dim * self.num_heads),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='edge_kernel',
            )
            self.bias_edge = self.add_weight(
                shape=(self.edge_dim* self.num_heads,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='edeg_bias',
            )

            self.kernel_wo = self.add_weight(
                shape=(self.num_heads * self.atom_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='proj_kernel',
            )
            self.bias_wo = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='proj_bias',
            )        

        self.leaky_relu = layers.LeakyReLU(alpha=0.2, name='leaky_relu')

        self.built = True
        
    
    def aggregate_nodes(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        atom_features_aggregated : TYPE
            DESCRIPTION.

        """
        atom_features, edges_features, pair_indices = inputs

        atom_features = tf.matmul(atom_features, self.kernel) + self.bias
        edges_features = tf.matmul(edges_features, self.kernel_edge) + self.bias_edge

        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_features_receive = atom_features_gather[:,0]
        atom_features_send = atom_features_gather[:,1]

        atom_features_receive = tf.reshape(atom_features_receive, shape=(-1, self.num_heads, self.atom_dim))
        atom_features_send = tf.reshape(atom_features_send, shape=(-1, self.num_heads, self.atom_dim))
        edges_features = tf.reshape(edges_features, shape=(-1, self.num_heads, self.edge_dim))

        atom_merge_features = tf.concat([atom_features_receive, atom_features_send, edges_features], axis=-1)

        a = tf.reduce_mean(atom_merge_features, axis=-1)
        activated_values = self.leaky_relu(a)
        exp = tf.exp(activated_values, name='exp')
        coeff_numerator = tf.repeat(exp, self.atom_dim, axis=-1)
        coeff_numerator = tf.reshape(coeff_numerator, shape=(-1, self.num_heads, self.atom_dim))
        coeff_denominator = tf.math.segment_sum(coeff_numerator, pair_indices[:,0])

        atom_features_updated = tf.multiply(coeff_numerator, atom_features_send)

        atom_features_aggregated = tf.math.segment_sum(atom_features_updated, pair_indices[:,0])
        atom_features_aggregated = tf.math.divide(atom_features_aggregated, coeff_denominator)
        atom_features_aggregated = tf.reshape(atom_features_aggregated, shape=(-1, self.num_heads * self.atom_dim))

        atom_features_aggregated = tf.matmul(atom_features_aggregated, self.kernel_wo) + self.bias_wo

        return atom_features_aggregated
    

    def call(self, inputs: Sequence) -> Sequence:
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        Sequence
            DESCRIPTION.

        """
        atom_features, edges_features, pair_indices = inputs

        atom_features_updated = atom_features
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edges_features, pair_indices])
            
        return atom_features_updated


    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        config.update({"steps": self.num_heads})
        return config
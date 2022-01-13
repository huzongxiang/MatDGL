# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:15:31 2021

@author: huzongxiang
"""


import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from crysnet.callbacks.cosineannealing import WarmUpCosineDecayScheduler
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


ModulePath = Path(__file__).parent.absolute()


class GNN:
    def __init__(self,
        model: Model,
        atom_dim=16,
        bond_dim=32,
        num_atom=118,
        state_dim=16,
        sp_dim=230,
        batch_size=16,
        regression=True,
        ntarget=1,
        multiclassification=None,
        optimizer='Adam',
        **kwargs,
        ):
        self.model = model
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_atom = num_atom
        self.state_dim = state_dim
        self.sp_dim = sp_dim
        self.batch_size = batch_size
        self.regression = regression
        self.ntarget = ntarget
        self.multiclassification = multiclassification
        self.optimizer = optimizer

        self.gnn = model(atom_dim=atom_dim,
        bond_dim=bond_dim,
        num_atom=num_atom,
        state_dim=state_dim,
        sp_dim=sp_dim,
        batch_size=batch_size,
        regression=regression,
        multiclassification=multiclassification,
        **kwargs)


    def __getattr__(self, attr):
        return getattr(self.gnn, attr)


    def train(self, train_data, valid_data=None, test_data=None, epochs=200, lr=1e-3, warm_up=True, warmrestart=None, load_weights=False, verbose=1, checkpoints=None, save_weights_only=True, workdir=None):
        
        gnn = self.gnn
        if self.regression:
            gnn.compile(
                loss=keras.losses.MeanAbsoluteError(),
                optimizer=self.optimizer,
                metrics=[keras.metrics.MeanAbsoluteError(name='mae')],
            )
        elif self.multiclassification:
            gnn.compile(
                loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer=self.optimizer,
                metrics=[tf.keras.metrics.AUC(name="AUC")],
            )
        else:
            gnn.compile(
                loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=self.optimizer,
                metrics=[tf.keras.metrics.AUC(name="AUC")],
            )

        print(gnn.summary())
        keras.utils.plot_model(gnn, Path(workdir/"gnn_arch.png"), show_dtype=True, show_shapes=True)

        if load_weights:
            print('load weights')
            path = train_data.task_type + ".hdf5"
            if load_weights == 'default':
                best_checkpoint = Path(ModulePath/"model"/path)
            elif load_weights == 'custom':
                best_checkpoint = Path(workdir/"model"/path)
            else:
                raise ValueError('load_weights should be "default" or "custom"')
            gnn.load_weights(best_checkpoint)
        print(train_data.task_type)
        Path(workdir/"model").mkdir(exist_ok=True)
        Path(workdir/"model"/train_data.task_type).mkdir(exist_ok=True)
        if checkpoints is None:
            if self.regression:
                filepath = Path(workdir/"model"/train_data.task_type/"gnn_{epoch:02d}-{val_mae:.3f}.hdf5")
                checkpoint = ModelCheckpoint(filepath, monitor='val_mae', save_best_only=True, save_weights_only=save_weights_only, verbose=verbose, mode='min')
            else:
                filepath = Path(workdir/"model"/train_data.task_type/"gnn_{epoch:02d}-{val_AUC:.3f}.hdf5")
                checkpoint = ModelCheckpoint(filepath, monitor='val_AUC', save_best_only=True, save_weights_only=save_weights_only, verbose=verbose, mode='max')

            earlystop = EarlyStopping(monitor='val_loss', patience=200, verbose=verbose, mode='min')

            if warm_up:
                sample_count = train_data.data_size
                warmup_epoch = 5
                train_per_epoch = sample_count / self.batch_size
                warmup_steps = warmup_epoch * train_per_epoch
                restart_epoches = warmrestart

                warm_up_lr = WarmUpCosineDecayScheduler(epochs=epochs,
                                                        restart_epoches=restart_epoches,
                                                        train_per_epoch=train_per_epoch,
                                                        learning_rate_base=lr,
                                                        warmup_learning_rate=2e-6,
                                                        warmup_steps=warmup_steps,
                                                        hold_base_rate_steps=5,
                                                        )

                checkpoints = [checkpoint, warm_up_lr, earlystop]
            else:
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=50, verbose=1, min_lr=1e-6, mode='min')
                checkpoints = [checkpoint, reduce_lr, earlystop]


        if valid_data:
            steps_per_train = int(np.ceil(train_data.data_size / self.batch_size))
            steps_per_val = int(np.ceil(valid_data.data_size / self.batch_size))
        else:
            steps_per_train = None
            steps_per_val = None

        print('gnn fit')
        history = gnn.fit(
            train_data,
            validation_data=valid_data,
            steps_per_epoch=steps_per_train,
            validation_steps=steps_per_val,
            epochs=epochs,
            verbose=verbose,
            callbacks=checkpoints,
            )

        Path(workdir/"results").mkdir(exist_ok=True)
        if self.regression:
            plot_train_regression(history, train_data.task_type, workdir)
            if test_data:
                plot_mae(gnn, test_data, workdir, name='test')
        else:
            plot_train(history, train_data.task_type, workdir)
            if test_data:
                if self.multiclassification:
                    plot_auc_multiclassification(gnn, test_data, self.multiclassification, workdir, name='test')
                else:
                    plot_auc(gnn, test_data, workdir, name='test')
        if warm_up:
            total_steps = int(epochs * sample_count / self.batch_size)
            plot_warm_up_lr(warm_up_lr, total_steps, lr, workdir)

        return gnn


    def predict_datas(self, test_data, workdir=None, load_weights='default'):
        print('load weights and predict...')
        save_file = test_data.task_type + ".hdf5"
        if load_weights:
            best_checkpoint = Path(ModulePath/"model"/save_file)
        else:
            best_checkpoint = Path(workdir/"model"/save_file)
        gnn = self.gnn()
        gnn.load_weights(best_checkpoint)
        Path(workdir/"results").mkdir(exist_ok=True)
        if self.regression:
            plot_mae(gnn, test_data, name='test')
        else:
            if self.multiclassification:
                plot_auc_multiclassification(gnn, test_data, self.multiclassification, workdir, name='test')
            else:
                plot_auc(gnn, test_data, name='test')      


    def predict(self, data, workdir=None, load_weights='default'):
        print('load weights and predict...')
        save_file = data.task_type + ".hdf5"
        if load_weights:
            best_checkpoint = Path(ModulePath/"model"/save_file)
        else:
            best_checkpoint = Path(workdir/"model"/save_file)
        gnn = self.gnn()
        gnn.load_weights(best_checkpoint)
        y_pred_keras = gnn.predict(data).ravel()
        return y_pred_keras


def plot_train(history, name, path):
    print('plot curve of training')
    plt.figure(figsize=(10, 12))
    plt.subplot(2,1,1)
    plt.plot(history.history["AUC"], label="train AUC")
    plt.plot(history.history["val_AUC"], label="valid AUC")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("AUC", fontsize=16)
    plt.legend(fontsize=16)
    plt.subplot(2,1,2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="valid loss")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("loss", fontsize=16)
    plt.legend(fontsize=16)
    save_path = name + "_train.png"
    plt.savefig(path/"results"/save_path)


def plot_train_regression(history, name, path):
    print('plot curve of training')
    plt.figure(figsize=(10, 12))
    plt.subplot(2,1,1)
    plt.plot(history.history["mae"], label="train mae")
    plt.plot(history.history["val_mae"], label="valid mae")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("mae", fontsize=16)
    plt.legend(fontsize=16)
    plt.subplot(2,1,2)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="valid loss")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("loss", fontsize=16)
    plt.legend(fontsize=16)
    save_path = name + "_train.png"
    plt.savefig(path/"results"/save_path)


def plot_auc(gnn, test_data, path, name='test'):
    print('predict')
    name = test_data.task_type + '_' + name
    y_pred_keras = gnn.predict(test_data).ravel()
    fpr_keras, tpr_keras, _ = roc_curve(test_data.labels, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve test')
    plt.legend(loc='best')
    save_path = name + "_predict" + ".png"
    plt.savefig(Path(path/"results"/save_path))


def plot_auc_multiclassification(gnn, test_data, n_classes, path, name='test'):
    print('predict')
    name = test_data.task_type + '_' + name
    y_pred_keras = gnn.predict(test_data)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_data.labels[:, i], y_pred_keras[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(np.array(test_data.labels)[:, i], y_pred_keras[:, i])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 6))
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    save_path = name + "_predict" + ".png"
    plt.savefig(Path(path/"results"/save_path))


def plot_mae(gnn, test_data, path, name='test'):
    print('predict')
    name = test_data.task_type + '_' + name
    y_pred_keras = gnn.predict(test_data).ravel()
    plt.figure(figsize=(10, 6))
    plt.scatter(test_data.labels, y_pred_keras)
    plt.plot([0, 8], [0, 8], 'k--')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel("experimetal", fontsize=16)
    plt.ylabel("pred", fontsize=16)
    plt.title('predicted')
    save_path = name + "_predict" + ".png"
    plt.savefig(Path(path/"results"/save_path))


def plot_warm_up_lr(warm_up_lr, total_steps, lr, path):
    plt.plot(warm_up_lr.learning_rates)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('lr', fontsize=20)
    plt.axis([0, total_steps, 0, lr*1.1])
    # plt.xticks(np.arange(0, epochs, 1))
    plt.grid()
    plt.title('Cosine decay with warmup', fontsize=20)
    plt.savefig(Path(path/"results"/"cosine_decay.png"))    

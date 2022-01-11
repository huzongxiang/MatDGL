# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:12:08 2021

@author: huzongxiang
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    Parameters
    ----------
    global_step : TYPE
        DESCRIPTION.
    learning_rate_base : TYPE
        DESCRIPTION.
    total_steps : TYPE
        DESCRIPTION.
    warmup_learning_rate : TYPE, optional
        DESCRIPTION. The default is 0.0.
    warmup_steps : TYPE, optional
        DESCRIPTION. The default is 0.
    hold_base_rate_steps : TYPE, optional
        DESCRIPTION. The default is 0.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if total_steps < warmup_steps:
         raise ValueError('total_steps must be larger or equal to '
                          'warmup_steps.')

    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps))) + 1e-7

    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 epochs=0,
                 restart_epoches=0,
                 train_per_epoch=0,
                 learning_rate_base=0,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super().__init__()
        self.restart_epoches = restart_epoches
        self.train_per_epoch = train_per_epoch
        self.learning_rate_base = learning_rate_base
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []
        self.former_epochs = 0
        self.steps = int(epochs * train_per_epoch)
        if restart_epoches:
            self.epochs_hold = restart_epoches
            self.steps = int(restart_epoches * train_per_epoch)
            self.cumsum = restart_epoches
            self.cumul = restart_epoches
    
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.restart_epoches:
            if epoch >= self.restart_epoches:
                self.global_step = 0
                self.former_epochs = self.restart_epoches
                self.cumsum = self.cumsum + self.cumul
                self.restart_epoches = self.restart_epoches + self.cumsum
                self.epochs_hold = self.restart_epoches - self.former_epochs
                self.steps = int(self.epochs_hold * self.train_per_epoch)


    def on_batch_end(self, batch, logs=None):
        """
        Parameters
        ----------
        batch : TYPE
            DESCRIPTION.
        logs : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    
    def on_batch_begin(self, batch, logs=None):
        """
        Parameters
        ----------
        batch : TYPE
            DESCRIPTION.
        logs : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))

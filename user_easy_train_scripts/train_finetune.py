# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:07:55 2022

@author: huzongxiang
"""


import warnings
import logging
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from matdgl.data import Dataset
from matdgl.models import Finetune
from matdgl.data.generator import GraphGenerator
from matdgl.models.finetune import FinetuneTransformer, FinetuneTransformerRes

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

np.random.seed(88)
tf.random.set_seed(88)

# get current work dir of test.py
ModulePath = Path(__file__).parent.absolute()

# read datas from ModulePath/datas/multiclassfication.json
print('reading dataset...')
'multiclassification'
'topology_multi'
'formation_energy'
'regression'
'my_regression'
start = time.time()
dataset = Dataset(task_type='topology_multi', data_path=ModulePath, ratio=[0.6, 0.8])
end = time.time()
run_time = end - start
print('done')
print('run time:  {:.2f} s'.format(run_time))
print(dataset.dataset_file)

BATCH_SIZE = 16
DATA_SIZE = None
CUTOFF = 4.5

# building batch generator for model trainning
Generators = GraphGenerator(dataset, data_size=DATA_SIZE, batch_size=BATCH_SIZE, cutoff=CUTOFF)
train_data = Generators.train_generator
valid_data = Generators.valid_generator
test_data = Generators.test_generator

#if task is multiclassfication, should define variable multiclassifiction
multiclassification = Generators.multiclassification 

epochs=32
lr=1e-3

# default trainning parameters
state_dim=16
sp_dim=230
output_dim=32
readout_units=128
dropout=0.0
reg2=0.0
reg3=0.0
reg_rec=0.0
regression=dataset.regression
optimizer = 'Adam'

print('\n----- parameters -----',
    '\ntask_type: ', dataset.task_type,
    '\nsample_size: ', Generators.data_size,
    '\ncutoff: ', CUTOFF,
    '\nstate_dim: ', state_dim,
    '\nsp_dim: ', sp_dim,
    '\noutput_dim: ', output_dim,
    '\nreadout_units: ', readout_units,
    '\ndropout: ', dropout,
    '\nreg2: ', reg2,
    '\nreg3: ', reg3,
    '\nreg_rec: ', reg_rec,
    '\noptimizer: ', optimizer,
    '\nmulticlassification: ', multiclassification,
    '\nregression: ', regression,)

del dataset

# default model is a GraphTransformer model, can be changed to MPNN model by set 'model=MpnnModel'
gnn = Finetune(model=FinetuneTransformerRes,
        state_dim=state_dim,
        sp_dim=sp_dim,
        output_dim=output_dim,
        readout_units=readout_units,
        dropout=dropout,
        reg2=reg2,
        reg3=reg3,
        reg_rec=reg_rec,
        optimizer = optimizer,
        regression=regression,
        multiclassification=multiclassification,
        )

# trainning model
gnn.train(train_data, valid_data, test_data, epochs=epochs, lr=lr, warm_up=True,\
             verbose=1, checkpoints=None, save_weights_only=True, workdir=ModulePath)

print('\n----- parameters -----',
    '\nsample_size: ', Generators.data_size,
    '\ncutoff: ', CUTOFF,
    '\nstate_dim: ', state_dim,
    '\nsp_dim: ', sp_dim,
    '\noutput_dim: ', output_dim,
    '\nreadout_units: ', readout_units,
    '\noptimizer: ', optimizer,
    '\nmulticlassification: ', multiclassification,
    '\nregression: ', regression,
    '\nepochs: ', epochs,
    '\nlr: ', lr)
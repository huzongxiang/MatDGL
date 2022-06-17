# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:09:46 2022

@author: hzx
"""

import warnings
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from matdgl.data import Dataset
from matdgl.models import Pretrainer
from matdgl.data.generator import GraphGeneratorMasking
from matdgl.models.pretrainer import TransformerModel 

tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

np.random.seed(52)
tf.random.set_seed(52)

# get current work dir of test.py
ModulePath = Path(__file__).parent.absolute()

# read datas from ModulePath/datas/multiclassfication.json
print('reading dataset...')
dataset = Dataset(task_type='pretrainning', data_path=ModulePath, ratio=[0.70, 0.99])
print('done')
print(dataset.dataset_file)

BATCH_SIZE = 32
DATA_SIZE = None
CUTOFF = 2.5

# building batch generator for model trainning
generators = GraphGeneratorMasking(dataset, data_size=DATA_SIZE, batch_size=BATCH_SIZE, cutoff=CUTOFF)
train_data, valid_data = generators()

# default trainning parameters
atom_dim=16
bond_dim=64
num_atom=119
state_dim=16
sp_dim=230
units=32
edge_steps=1
transform_steps=1
num_attention_heads=8
dense_units=64
reg0=0.00
reg1=0.00
reg2=0.00
batch_size=BATCH_SIZE
spherical_harmonics=True
optimizer = 'Adam'

print('\n----- parameters -----',
    '\ntask_type: ', dataset.task_type,
    '\nsample_size: ', generators.data_size,
    '\ncutoff: ', CUTOFF,
    '\natom_dim: ', atom_dim,
    '\nbond_dim: ', bond_dim,
    '\nnum_atom: ', num_atom,
    '\nstate_dim: ', state_dim,
    '\nsp_dim: ', sp_dim,
    '\nunits: ', units,
    '\nedge_steps: ', edge_steps,
    '\ntransform_steps: ', transform_steps,
    '\nnum_attention_heads: ', num_attention_heads,
    '\ndense_units: ', dense_units,
    '\nreg0: ', reg0,
    '\nreg1: ', reg1,
    '\nreg2: ', reg2,
    '\nbatch_size: ', batch_size,
    '\nspherical_harmonics: ', spherical_harmonics,
    '\noptimizer: ', optimizer,
    )

del dataset

# default model is a GraphTransformer model, can be changed to MPNN model by set 'model=MpnnModel'
gnn = Pretrainer(model=TransformerModel,
        atom_dim=atom_dim,
        bond_dim=bond_dim,
        num_atom=num_atom,
        state_dim=state_dim,
        sp_dim=sp_dim,
        units=units,
        edge_steps=edge_steps,
        transform_steps=transform_steps,
        num_attention_heads=num_attention_heads,
        dense_units=dense_units,
        reg0=reg0,
        reg1=reg1,
        batch_size=batch_size,
        spherical_harmonics=spherical_harmonics,
        optimizer = optimizer,
        )

# trainning model
gnn.train(train_data, valid_data, test_data=None, epochs=10, lr=1e-4, warm_up=True,
        load_weights=False, verbose=1, checkpoints=None, save_weights_only=True, workdir=ModulePath)
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:17:16 2022

@author: huzongxiang
"""


from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from matdgl.layers import Set2Set
from matdgl.models.pretrainer import TransformerModel


ModulePath = Path(__file__).parent.absolute()


def FinetuneTransformer(state_dim=16,
                        sp_dim=230,
                        output_dim=32,
                        readout_units=128,
                        dropout=0.0,
                        reg2=0.0,
                        reg3=0.0,
                        reg_rec=0.0,
                        regression=False,
                        ntarget=1,
                        multiclassification=None,
                        weight_path=Path(ModulePath/"model/transformer.hdf5"),
                        ):


    transformer = TransformerModel(atom_dim=16,
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
                                    spherical_harmonics=True)

    transformer.load_weights(weight_path)

    for layer in transformer.layers:
        layer.trainable = False

    x_nodes, edges_matrixs = transformer.layers[-5].output
    state_attrs = transformer.layers[-2].output
    state_attrs_ = layers.Embedding(sp_dim, state_dim, dtype="float32", name="state_attrs")(state_attrs)

    x_state = layers.Dense(16, kernel_regularizer=l2(reg2))(state_attrs_)

    x_node = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
    x_edge = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(edges_matrixs, edge_mode=True)
    
    x = layers.Concatenate(axis=-1, name='concat')([x_node, x_edge, x_state])

    # x = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x)
        
    x = layers.Dense(readout_units, activation="relu", kernel_regularizer=l2(reg3), name='readout0')(x)
    
    x_orgin = x

    x = layers.Dense(readout_units, activation="relu", kernel_regularizer=l2(reg3), name='res0')(x)
    x = layers.Dense(readout_units//2, activation="relu", kernel_regularizer=l2(reg3), name='res1')(x)
    x = layers.Dense(readout_units//4, activation="relu", kernel_regularizer=l2(reg3), name='res2')(x)
    x = layers.Dense(readout_units//2, activation="relu", kernel_regularizer=l2(reg3), name='res3')(x)
    x = layers.Dense(readout_units, activation="relu", kernel_regularizer=l2(reg3), name='res4')(x)

    x = layers.Add()([x, x_orgin])

    if dropout:
        x = layers.Dropout(dropout, name='dropout0')(x)

    x = layers.Dense(readout_units//2, activation="relu", kernel_regularizer=l2(reg3), name='readout1')(x)

    if dropout:
        x = layers.Dropout(dropout, name='dropout1')(x)

    x = layers.Dense(readout_units//4, activation="relu", kernel_regularizer=l2(reg3), name='readout2')(x)

    if dropout:
        x = layers.Dropout(dropout, name='dropout')(x)

    if regression:
        x = layers.Dense(ntarget, name='final')(x)
    elif multiclassification is not None:
        x = layers.Dense(multiclassification, activation="softmax", name='final_softmax')(x)
    else:
        x = layers.Dense(1, activation="sigmoid", name='final')(x)

    model = Model(
        inputs=transformer.input[:-2],
        outputs=[x],
    )
    return model


def FinetuneTransformer1(state_dim=16,
                        sp_dim=230,
                        output_dim=32,
                        readout_units=128,
                        dropout=0.0,
                        reg2=0.0,
                        reg3=0.0,
                        reg_rec=0.0,
                        regression=False,
                        ntarget=1,
                        multiclassification=None,
                        weight_path=Path(ModulePath/"model/transformer.hdf5"),
                        ):


    transformer = TransformerModel(atom_dim=16,
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
                                    spherical_harmonics=True)

    transformer.load_weights(weight_path)

    for layer in transformer.layers:
        layer.trainable = False

    x_nodes, edges_matrixs = transformer.layers[-5].output
    state_attrs = transformer.layers[-2].output
    state_attrs_ = layers.Embedding(sp_dim, state_dim, dtype="float32", name="state_attrs")(state_attrs)

    x_state = layers.Dense(16, kernel_regularizer=l2(reg2))(state_attrs_)

    x_node = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
    x_edge = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(edges_matrixs, edge_mode=True)
    
    x = layers.Concatenate(axis=-1, name='concat')([x_node, x_edge, x_state])

    x = layers.Dense(readout_units, activation="relu", kernel_regularizer=l2(reg3), name='readout0')(x)
    
    if dropout:
        x = layers.Dropout(dropout, name='dropout0')(x)

    x = layers.Dense(readout_units//2, activation="relu", kernel_regularizer=l2(reg3), name='readout1')(x)

    if dropout:
        x = layers.Dropout(dropout, name='dropout1')(x)

    x = layers.Dense(readout_units//4, activation="relu", kernel_regularizer=l2(reg3), name='readout2')(x)

    if dropout:
        x = layers.Dropout(dropout, name='dropout')(x)

    if regression:
        x = layers.Dense(ntarget, name='final')(x)
    elif multiclassification is not None:
        x = layers.Dense(multiclassification, activation="softmax", name='final_softmax')(x)
    else:
        x = layers.Dense(1, activation="sigmoid", name='final')(x)

    model = Model(
        inputs=transformer.input[:-2],
        outputs=[x],
    )
    return model
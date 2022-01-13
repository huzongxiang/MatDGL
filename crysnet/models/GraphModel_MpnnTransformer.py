# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:19:27 2021

@author: huzongxiang
"""


from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from EdgeNetworkLayer import EdgeMessagePassing
from PartitionPaddingLayer import PartitionPadding
from MPNNTransformer import MpnnTransformerEncoder
from Readout import Set2Set


def GraphformerModel(
        bond_dim,
        atom_dim=16,
        num_atom=118,
        state_dim=16,
        sp_dim=230,
        units=32,
        edge_steps=1,
        message_steps=1,
        transform_steps=1,
        num_attention_heads=8,
        dense_units=32,
        output_dim=32,
        readout_units=128,
        dropout=0.0,
        reg0=0.0,
        reg1=0.0,
        reg2=0.0,
        reg3=0.0,
        reg_rec=0.0,
        batch_size=16,
        spherical_harmonics=False,
        regression=False,
        multiclassification=None,
        ):
        atom_features = layers.Input((), dtype="int32", name="atom_features_input")
        atom_features_ = layers.Embedding(num_atom, atom_dim, dtype="float32", name="atom_features")(atom_features)
        bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
        local_env = layers.Input((6), dtype="float32", name="local_env")
        state_attrs = layers.Input((), dtype="int32", name="state_attrs_input")   
        state_attrs_ = layers.Embedding(sp_dim, state_dim, dtype="float32", name="state_attrs")(state_attrs)

        pair_indices = layers.Input((2), dtype="int32", name="pair_indices")

        atom_graph_indices = layers.Input(
            (), dtype="int32", name="atom_graph_indices"
        )

        bond_graph_indices = layers.Input(
            (), dtype="int32", name="bond_graph_indices"
        )

        pair_indices_per_graph = layers.Input((2), dtype="int32", name="pair_indices_per_graph")

        x_nodes_, x_edges_, x_state = MpnnTransformerEncoder()([atom_features_, bond_features, state_attrs_, pair_indices, atom_graph_indices, bond_graph_indices])
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)

        x_node = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
        x_edge = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_edges)
        
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
            x = layers.Dense(1, name='final')(x)
        elif multiclassification is not None:
            x = layers.Dense(multiclassification, activation="softmax", name='final_softmax')(x)
        else:
            x = layers.Dense(1, activation="sigmoid", name='final')(x)

        model = Model(
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model
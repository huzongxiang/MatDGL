# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:14:40 2022

@author: huzongxiang
"""

from typing import final
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from matdgl.layers import MessagePassing, NewMessagePassing
from matdgl.layers import SphericalBasisLayer, AzimuthLayer, ConcatLayer, EdgeMessagePassing
from matdgl.layers import PartitionPadding, PartitionPaddingPair
from matdgl.layers import EdgesAugmentedLayer, GraphTransformerEncoder
from matdgl.layers import GNGroverEncoder, ConvGroverEncoder
from matdgl.layers import GraphAttentionLayer
from matdgl.layers import CrystalGraphConvolution
from matdgl.layers import Set2Set, LinearPredMasking


def TransformerModel(
        bond_dim,
        atom_dim=16,
        num_atom=118,
        state_dim=16,
        sp_dim=230,
        units=32,
        edge_steps=1,
        transform_steps=1,
        num_attention_heads=8,
        dense_units=32,
        reg0=0.0,
        reg1=0.0,
        batch_size=16,
        spherical_harmonics=False,
        final_dim=119,
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

        masking_indices = layers.Input((), dtype="int32", name="masking_indices")
        masking_graph_indices = layers.Input((), dtype="int32", name="masking_graph_indices")

        x_nodes_ = layers.Dense(16, kernel_regularizer=l2(reg0))(atom_features_)

        x_edges_ = EdgeMessagePassing(units,
                                        edge_steps,
                                        kernel_regularizer=l2(reg0),
                                        sph=spherical_harmonics
                                        )([bond_features, local_env, pair_indices])

        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)

        pair_indices_batch = PartitionPaddingPair(batch_size)([pair_indices_per_graph, bond_graph_indices])
        
        edges_matrixs_a = EdgesAugmentedLayer()([x_edges, pair_indices_batch])
        
        x_nodes_a = layers.Masking(mask_value=0.)(x_nodes)

        x_nodes = x_nodes_a
        edges_matrixs = edges_matrixs_a

        for i in range(transform_steps):
            x_nodes, edges_matrixs = GraphTransformerEncoder(
                num_attention_heads,
                atom_dim,
                bond_dim,
                dense_units,
                kernel_regularizer=l2(reg1),
                )([x_nodes, edges_matrixs])

        x = LinearPredMasking(units=final_dim)([x_nodes, masking_indices, masking_graph_indices])

        model = Model(
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph, masking_indices, masking_graph_indices],
            outputs=[x],
        )
        return model
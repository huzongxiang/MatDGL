# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:19:27 2021

@author: huzongxiang
"""

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from crysnet.layers import MessagePassing, NewMessagePassing
from crysnet.layers import SphericalBasisLayer, AzimuthLayer, ConcatLayer, EdgeMessagePassing
from crysnet.layers import PartitionPadding, PartitionPaddingPair
from crysnet.layers import EdgesAugmentedLayer, GraphTransformerEncoder
from crysnet.layers import GraphormerEncoder, ConvGraphormerEncoder
from crysnet.layers import CrystalGraphConvolution
from crysnet.layers import Set2Set


def GraphModel(
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
        ntarget=1,
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

        x = MessagePassing(message_steps,
                            kernel_regularizer=l2(reg0),
                            recurrent_regularizer=l2(reg_rec),
                            )([atom_features_, bond_features, state_attrs_, pair_indices,
                                atom_graph_indices, bond_graph_indices]
                            )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]

        x_edges_ = EdgeMessagePassing(units,
                                        edge_steps,
                                        kernel_regularizer=l2(reg0),
                                        sph=spherical_harmonics
                                        )([x_edges_, local_env, pair_indices])
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)

        pair_indices_batch = PartitionPaddingPair(batch_size)([pair_indices_per_graph, bond_graph_indices])

        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)

        edges_matrixs = EdgesAugmentedLayer()([x_edges, pair_indices_batch])

        for i in range(transform_steps):
            x_nodes, edges_matrixs = GraphTransformerEncoder(
                num_attention_heads,
                atom_dim,
                bond_dim,
                dense_units,
                kernel_regularizer=l2(reg1),
                )([x_nodes, edges_matrixs])

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
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model


def MpnnModel(
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
        ntarget=1,
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

        edge_features = EdgeMessagePassing(units, edge_steps, sph=spherical_harmonics)([bond_features, local_env, pair_indices])

        x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, edge_features, state_attrs_, pair_indices,
             atom_graph_indices, bond_graph_indices]
        )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)
        
        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)

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
            x = layers.Dense(ntarget, name='final')(x)
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


def MpnnBaseModel(
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
        ntarget=1,
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

        x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, bond_features, state_attrs_, pair_indices,
             atom_graph_indices, bond_graph_indices]
        )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)
        
        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)

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
            x = layers.Dense(ntarget, name='final')(x)
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


def NewMpnnBaseModel(
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
        ntarget=1,
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

        x = NewMessagePassing(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, bond_features, state_attrs_, pair_indices,
             atom_graph_indices, bond_graph_indices]
        )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)
        
        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)

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
            x = layers.Dense(ntarget, name='final')(x)
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


def DirectionalMpnnModel(
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
        ntarget=1,
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

        if spherical_harmonics:
            sph_harm_features, edges_neighbor = SphericalBasisLayer(units=bond_dim, num_spherical=6)([local_env, pair_indices])
        else:
            sph_harm_features, edges_neighbor = AzimuthLayer(units=bond_dim)([local_env, pair_indices])
            
        edge_features = ConcatLayer(units=bond_dim)([bond_features, sph_harm_features, edges_neighbor])

        x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, edge_features, state_attrs_, pair_indices,
             atom_graph_indices, bond_graph_indices]
        )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)
        
        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)

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
            x = layers.Dense(ntarget, name='final')(x)
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


def TransformerModel(
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
        ntarget=1,
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

        x_nodes_ = layers.Dense(16, kernel_regularizer=l2(reg0))(atom_features_)
        x_state = layers.Dense(16, kernel_regularizer=l2(reg0))(state_attrs_)

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
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model


def TransformerBaseModel(
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
        ntarget=1,
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

        x_nodes_ = layers.Dense(16, kernel_regularizer=l2(reg0))(atom_features_)
        x_edges_ = layers.Dense(64, kernel_regularizer=l2(reg0))(bond_features)
        x_state = layers.Dense(16, kernel_regularizer=l2(reg0))(state_attrs_)

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
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model


def DirectionalTransformerModel(
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
        ntarget=1,
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

        x_nodes_ = layers.Dense(16, kernel_regularizer=l2(reg0))(atom_features_)
        x_state = layers.Dense(16, kernel_regularizer=l2(reg0))(state_attrs_)

        if spherical_harmonics:
            sph_harm_features, edges_neighbor = SphericalBasisLayer(units=bond_dim, num_spherical=6)([local_env, pair_indices])
        else:
            sph_harm_features, edges_neighbor = AzimuthLayer(units=bond_dim)([local_env, pair_indices])
            
        x_edges_ = ConcatLayer(units=bond_dim)([bond_features, sph_harm_features, edges_neighbor])

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
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model


def CgcnnModel(
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
        ntarget=1,
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

        x = CrystalGraphConvolution(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, bond_features, pair_indices]
        )
    
        x = PartitionPadding(batch_size)([x, atom_graph_indices])

        x = layers.BatchNormalization()(x)

        x = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x)
            
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
            inputs=[atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices,
                    bond_graph_indices, pair_indices_per_graph],
            outputs=[x],
        )
        return model


def GraphormerModel(
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

        x_nodes_, x_edges_, x_state = GraphormerEncoder()([atom_features_, bond_features, state_attrs_, pair_indices, atom_graph_indices, bond_graph_indices])
      
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


def ConvGraphormerModel(
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

        x_nodes_ = ConvGraphormerEncoder()([atom_features_, bond_features,
                                         state_attrs_, pair_indices, atom_graph_indices, bond_graph_indices])
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)

        x = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
            
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
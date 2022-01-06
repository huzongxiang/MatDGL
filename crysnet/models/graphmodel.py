# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:19:27 2021

@author: hzx
"""

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from crysnet.layers import MessagePassing
from crysnet.layers import EdgeMessagePassing
from crysnet.layers import PartitionPadding, PartitionPaddingPair
from crysnet.layers import EdgesAugmentedLayer, GraphTransformerEncoder
from crysnet.layers import Set2Set


def GraphModel(
        bond_dim,
        atom_dim=16,
        num_atom=118,
        state_dim=16,
        sp_dim=230,
        units=32,
        edge_steps=1,
        message_steps=5,
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

        # edge_features = EdgeMessagePassing(units,
        #                                 edge_steps,
        #                                 kernel_regularizer=l2(reg0),
        #                                 sph=spherical_harmonics
        #                                 )([bond_features, local_env, pair_indices])

        x = MessagePassing(message_steps,
                            kernel_regularizer=l2(reg0),
                            recurrent_regularizer=l2(reg_rec),
                            )([atom_features_, bond_features, state_attrs_, pair_indices,
                                atom_graph_indices, bond_graph_indices]
                            )

        # x = MessagePassing(message_steps,
        #                     kernel_regularizer=l2(reg0),
        #                     recurrent_regularizer=l2(reg0),
        #                     )([atom_features_, edge_features, state_attrs_, pair_indices,
        #                         atom_graph_indices, bond_graph_indices]
        #                     )
        
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
        
        # x_nodes = layers.Masking(mask_value=0.)(x_nodes_b)
        # x_edges = layers.Masking(mask_value=0.)(x_edges_b)

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
        
        # x_nodes = layers.BatchNormalization()(x_nodes)
        # edges_matrixs = layers.BatchNormalization()(edges_matrixs)

        # x_nodes_ = layers.Add()([x_nodes_a, x_nodes])
        # edges_matrixs_ = layers.Add()([edges_matrixs_a, edges_matrixs])

        # x_nodes = layers.Activation(activation="relu")(x_nodes_)
        # edges_matrixs_ = layers.Activation(activation="relu")(edges_matrixs_)

        x_node = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
        x_edge = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(edges_matrixs, edge_mode=True)

        # x_nodes = TransformerEncoder(num_attention_heads, embed_dim=16, dense_dim=32)(x_nodes)
        # x_edges = TransformerEncoder(num_attention_heads,  embed_dim=16, dense_dim=32)(x_edges)

        # x_node = layers.GlobalAveragePooling1D()(x_nodes)
        # x_edge = layers.GlobalAveragePooling1D()(x_edges)
        
        x = layers.Concatenate(axis=-1, name='concat')([x_node, x_edge, x_state])
            
        # x = layers.BatchNormalization()(x)   

        x = layers.Dense(readout_units, activation="relu", kernel_regularizer=l2(reg3), name='readout0')(x)

        # x = layers.BatchNormalization()(x) 

        if dropout:
            x = layers.Dropout(dropout, name='dropout0')(x)

        x = layers.Dense(readout_units//2, activation="relu", kernel_regularizer=l2(reg3), name='readout1')(x)

        # x = layers.BatchNormalization()(x) 

        if dropout:
            x = layers.Dropout(dropout, name='dropout1')(x)

        x = layers.Dense(readout_units//4, activation="relu", kernel_regularizer=l2(reg3), name='readout2')(x)
        
        # x = layers.BatchNormalization()(x)   

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

        # x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
        #     [atom_features_, bond_features, state_attrs_, pair_indices,
        #      atom_graph_indices, bond_graph_indices]
        # )

        x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
            [atom_features_, edge_features, state_attrs_, pair_indices,
             atom_graph_indices, bond_graph_indices]
        )
        
        x_nodes_ = x[0]
        x_edges_ = x[1]
        x_state = x[2]
        
        # edge_features = EdgeMessagePassing(units)([x_edges_, local_env, pair_indices])
      
        # x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        # x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])


        # x_edges_ = EdgeMessagePassing(units)([x_edges_, local_env, pair_indices])
      
        x_nodes_b = PartitionPadding(batch_size)([x_nodes_, atom_graph_indices])
        x_edges_b = PartitionPadding(batch_size)([x_edges_, bond_graph_indices])

        x_nodes = layers.BatchNormalization()(x_nodes_b)
        x_edges = layers.BatchNormalization()(x_edges_b)
        
        # edges_matrixs_a = EdgesAugmentedLayer()([x_edges, pair_indices_batch])
        
        x_nodes = layers.Masking(mask_value=0.)(x_nodes)
        x_edges = layers.Masking(mask_value=0.)(x_edges)
        # # edges_matrixs = Masking(mask_value=0.)(edges_matrixs_)  

        # x_nodes = x_nodes_a
        # edges_matrixs = edges_matrixs_a

        # for i in range(transform_steps):
        #     x_nodes, edges_matrixs = GraphTransformerEncoder(
        #         num_attention_heads,
        #         atom_dim,
        #         bond_dim,
        #         dense_units,
        #         kernel_regularizer=l2(reg0),
        #         )([x_nodes, edges_matrixs])
        
        # x_node_ = layers.BatchNormalization()(x_nodes)
        # edges_matrixs_ = layers.BatchNormalization()(edges_matrixs)

        # x_nodes_ = layers.Add()([x_nodes_a, x_nodes])
        # edges_matrixs_ = layers.Add()([edges_matrixs_a, edges_matrixs])

        # x_nodes = layers.Activation(activation="relu")(x_nodes_)
        # edges_matrixs_ = layers.Activation(activation="relu")(edges_matrixs_)

        x_node = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_nodes)
        x_edge = Set2Set(output_dim, kernel_regularizer=l2(reg2), recurrent_regularizer=l2(reg_rec))(x_edges)

        # x_nodes = TransformerEncoder(num_attention_heads, embed_dim=16, dense_dim=32)(x_nodes)
        # x_edges = TransformerEncoder(num_attention_heads,  embed_dim=16, dense_dim=32)(x_edges)

        # x_node = layers.GlobalAveragePooling1D()(x_nodes)
        # x_edge = layers.GlobalAveragePooling1D()(x_edges)
        
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


def TransformerModel(
        bond_dim,
        atom_dim=16,
        num_atom=118,
        state_dim=16,
        sp_dim=230,
        units=32,
        edge_steps=1,
        message_steps=5,
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

        # edge_features = EdgeMessagePassing(units)([bond_features, local_env, pair_indices])

        # x = MessagePassing(message_steps, kernel_regularizer=l2(reg0))(
        #     [atom_features_, bond_features, state_attrs_, pair_indices,
        #      atom_graph_indices, bond_graph_indices]
        # )

        x_nodes_ = layers.Dense(16, kernel_regularizer=l2(reg0))(atom_features_)
        # x_edges_ = layers.Dense(64, kernel_regularizer=l2(reg0))(bond_features)
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
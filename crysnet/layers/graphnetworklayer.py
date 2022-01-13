# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:33:33 2021

@author: huzongxiang
"""

from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints


class MessagePassing(layers.Layer):
    def __init__(self,
        steps,
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
        self.state_dim = input_shape[2][-1]

        self.update_nodes = layers.GRUCell(self.atom_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_nodes'
                                            )

        self.update_edges = layers.GRUCell(self.edge_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_edges'
                                            )

        self.update_states = layers.GRUCell(self.state_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_states')
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel = self.add_weight(
                shape=(self.edge_dim, 
                       (self.atom_dim + self.state_dim)**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel',
            )
            self.bias = self.add_weight(
                shape=((self.atom_dim + self.state_dim)**2,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias',
            )
        
        self.built = True


    def concat_edges(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # each bond in pair_indices, concatenate thier atom features by atom indexes
        # then concatenate atom features with bond features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        edges_merge_atom_features = tf.concat([atom_features_gather[:,0], atom_features_gather[:,1]], axis=-1)
        
        # repeat state attributes by bond_graph_indices, then concatenate to bond_merge_atom_features
        state_attrs_repeat = tf.gather(state_attrs, bond_graph_indices)
        edges_features_concated = tf.concat([edges_merge_atom_features, state_attrs_repeat, edges_features], axis=-1)

        return edges_features_concated
        
    
    def aggregate_nodes(self, inputs: Sequence):
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
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        state_attrs_repeat = tf.gather(state_attrs, atom_graph_indices)
        atom_merge_state_features = tf.concat([atom_features, state_attrs_repeat], axis=-1)
        
        # using bond_updated to update atom_features_neighbors
        # using num_bonds of bond feature to renew num_bonds of adjacent atom feature
        # bond feature with shape (bond_dim,), not a Matrix, multiply by a learnable weight
        # with shape (atom_dim,atom_dim,bond_dim), then bond feature transfer to shape (atom_dim,atom_dim)
        # the bond matrix with shape (atom_dim,atom_dim) can update atom_feature with shape (aotm_dim,)
        # so num_bonds of bond features need num_bonds of bond matrix, so a matrix with shape
        # (num_bonds,(atom_dim,atom_dim,bond_dim)) to transfer bond_features to shape (num_bonds,(atom_dim,atom_dim))
        # finally, apply this bond_matrix to adjacent atoms, get bond_features updated atom_features_neighbors
        edges_weights = tf.matmul(edges_features, self.kernel) + self.bias
        edges_weights = tf.reshape(edges_weights, (-1, self.atom_dim + self.state_dim, self.atom_dim + self.state_dim))
        atom_features_neighbors = tf.gather(atom_merge_state_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(edges_weights, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        
        # using conbination of tf.operation realizes multiplicationf between adjacent matrix and atom features
        # first tf.gather end features using end atom index pair_indices[:,1] to atom_features_neighbors
        # then using bond matrix updates atom_features_neighbors, get transformed_features
        # finally tf.segment_sum calculates sum of updated neighbors feature by start atom index pair_indices[:,0]
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:,0])

        return atom_features_aggregated
        
    
    def concat_states(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        state_attrs_aggregated : TYPE
            DESCRIPTION.

        """
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # concat state_attrs with bond_updated and atom_features_aggregated
        edges_features_sum = tf.math.segment_sum(edges_features, bond_graph_indices)
        atom_features_sum = tf.math.segment_sum(atom_features, atom_graph_indices)
        state_attrs_concated = tf.concat([atom_features_sum, edges_features_sum, state_attrs], axis=-1)

        return state_attrs_concated


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
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        atom_features_updated = atom_features
        edges_features_updated =  edges_features
        state_attrs_updated = state_attrs

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # concate and update bond_features
            edges_features_concated = self.concat_edges(
                            [atom_features_updated, edges_features_updated,
                            state_attrs_updated, pair_indices, 
                            atom_graph_indices, bond_graph_indices]
                            )
            
            # Update edge_features via a step of GRU
            edges_features_updated, _ = self.update_edges(edges_features_concated, edges_features_updated)
            
            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.aggregate_nodes(
                                    [atom_features_updated, edges_features_updated,
                                    state_attrs_updated, pair_indices,
                                    atom_graph_indices, bond_graph_indices]
                                    )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_nodes(atom_features_aggregated, atom_features_updated)

            # update state_attrs
            state_attrs_concated = self.concat_states(
                                [atom_features_updated, edges_features_updated,
                                 state_attrs_updated, pair_indices,
                                 atom_graph_indices, bond_graph_indices]
                                )

            state_attrs_updated, _ = self.update_states(state_attrs_concated, state_attrs_updated)
            
        return [atom_features_updated, edges_features_updated, state_attrs_updated]
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config


class NewMessagePassing(layers.Layer):
    """
    Introducing a kernel sigmoid(node_features) * softplus(node_features) from Xie et al. PHYSICAL REVIEW LETTERS 120, 145301 (2018)
    """
    def __init__(self,
        steps,
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
        self.state_dim = input_shape[2][-1]

        self.update_nodes = layers.GRUCell(self.atom_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_nodes'
                                            )

        self.update_edges = layers.GRUCell(self.edge_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_edges'
                                            )

        self.update_states = layers.GRUCell(self.state_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_states')
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel_s = self.add_weight(
                shape=(self.edge_dim, 
                       (self.atom_dim + self.state_dim)**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_s',
            )
            self.bias_s = self.add_weight(
                shape=((self.atom_dim + self.state_dim)**2,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_s',
            )

            self.kernel_g = self.add_weight(
                shape=(self.edge_dim, 
                       (self.atom_dim + self.state_dim)**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_g',
            )
            self.bias_g = self.add_weight(
                shape=((self.atom_dim + self.state_dim)**2,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_g',
            )
        
        self.built = True


    def concat_edges(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # each bond in pair_indices, concatenate thier atom features by atom indexes
        # then concatenate atom features with bond features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        edges_merge_atom_features = tf.concat([atom_features_gather[:,0], atom_features_gather[:,1]], axis=-1)
        
        # repeat state attributes by bond_graph_indices, then concatenate to bond_merge_atom_features
        state_attrs_repeat = tf.gather(state_attrs, bond_graph_indices)
        edges_features_concated = tf.concat([edges_merge_atom_features, state_attrs_repeat, edges_features], axis=-1)

        return edges_features_concated
        
    
    def aggregate_nodes(self, inputs: Sequence):
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
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        state_attrs_repeat = tf.gather(state_attrs, atom_graph_indices)
        atom_merge_state_features = tf.concat([atom_features, state_attrs_repeat], axis=-1)
        
        # using bond_updated to update atom_features_neighbors
        # using num_bonds of bond feature to renew num_bonds of adjacent atom feature
        # bond feature with shape (bond_dim,), not a Matrix, multiply by a learnable weight
        # with shape (atom_dim,atom_dim,bond_dim), then bond feature transfer to shape (atom_dim,atom_dim)
        # the bond matrix with shape (atom_dim,atom_dim) can update atom_feature with shape (aotm_dim,)
        # so num_bonds of bond features need num_bonds of bond matrix, so a matrix with shape
        # (num_bonds,(atom_dim,atom_dim,bond_dim)) to transfer bond_features to shape (num_bonds,(atom_dim,atom_dim))
        # finally, apply this bond_matrix to adjacent atoms, get bond_features updated atom_features_neighbors
        edges_weights_s = tf.matmul(edges_features, self.kernel_s) + self.bias_s
        edges_weights_s = tf.reshape(edges_weights_s, (-1, self.atom_dim + self.state_dim, self.atom_dim + self.state_dim))

        edges_weights_g = tf.matmul(edges_features, self.kernel_g) + self.bias_g
        edges_weights_g = tf.reshape(edges_weights_g, (-1, self.atom_dim + self.state_dim, self.atom_dim + self.state_dim))

        atom_features_neighbors = tf.gather(atom_merge_state_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        transformed_features_s = tf.matmul(edges_weights_s, atom_features_neighbors)
        transformed_features_s = tf.squeeze(transformed_features_s, axis=-1)

        transformed_features_g = tf.matmul(edges_weights_g, atom_features_neighbors)
        transformed_features_g = tf.squeeze(transformed_features_g, axis=-1)
        
        # using conbination of tf.operation realizes multiplicationf between adjacent matrix and atom features
        # first tf.gather end features using end atom index pair_indices[:,1] to atom_features_neighbors
        # then using bond matrix updates atom_features_neighbors, get transformed_features
        # finally tf.segment_sum calculates sum of updated neighbors feature by start atom index pair_indices[:,0]
        transformed_features = tf.sigmoid(transformed_features_s) * tf.nn.softplus(transformed_features_g)
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:,0])

        return atom_features_aggregated
        
    
    def concat_states(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        state_attrs_aggregated : TYPE
            DESCRIPTION.

        """
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # concat state_attrs with bond_updated and atom_features_aggregated
        edges_features_sum = tf.math.segment_sum(edges_features, bond_graph_indices)
        atom_features_sum = tf.math.segment_sum(atom_features, atom_graph_indices)
        state_attrs_concated = tf.concat([atom_features_sum, edges_features_sum, state_attrs], axis=-1)

        return state_attrs_concated


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
        atom_features, edges_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        atom_features_updated = atom_features
        edges_features_updated =  edges_features
        state_attrs_updated = state_attrs

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # concate and update bond_features
            edges_features_concated = self.concat_edges(
                            [atom_features_updated, edges_features_updated,
                            state_attrs_updated, pair_indices, 
                            atom_graph_indices, bond_graph_indices]
                            )
            
            # Update edge_features via a step of GRU
            edges_features_updated, _ = self.update_edges(edges_features_concated, edges_features_updated)
            
            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.aggregate_nodes(
                                    [atom_features_updated, edges_features_updated,
                                    state_attrs_updated, pair_indices,
                                    atom_graph_indices, bond_graph_indices]
                                    )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_nodes(atom_features_aggregated, atom_features_updated)

            # update state_attrs
            state_attrs_concated = self.concat_states(
                                [atom_features_updated, edges_features_updated,
                                 state_attrs_updated, pair_indices,
                                 atom_graph_indices, bond_graph_indices]
                                )

            state_attrs_updated, _ = self.update_states(state_attrs_concated, state_attrs_updated)
            
        return [atom_features_updated, edges_features_updated, state_attrs_updated]
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config


class DirecitonalMessagePassing(layers.Layer):
    def __init__(self,
        steps,
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
        self.state_dim = input_shape[2][-1]

        self.update_nodes = layers.GRUCell(self.atom_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_nodes'
                                            )

        self.update_edges = layers.GRUCell(self.edge_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_edges'
                                            )

        self.update_states = layers.GRUCell(self.state_dim,
                                            kernel_regularizer=self.recurrent_regularizer,
                                            recurrent_regularizer=self.recurrent_regularizer,
                                            name='update_states')
        
        with tf.name_scope("nodes_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel = self.add_weight(
                shape=(self.edge_dim, 
                       (self.atom_dim + self.state_dim)**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel',
            )
            self.bias = self.add_weight(
                shape=((self.atom_dim + self.state_dim)**2,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias',
            )
        
        self.built = True


    def concat_edges(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        atom_features, edges_sph_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # each bond in pair_indices, concatenate thier atom features by atom indexes
        # then concatenate atom features with bond features
        atom_features_gather = tf.gather(atom_features, pair_indices)
        edges_merge_atom_features = tf.concat([atom_features_gather[:,0], atom_features_gather[:,1]], axis=-1)
        
        # repeat state attributes by bond_graph_indices, then concatenate to bond_merge_atom_features
        state_attrs_repeat = tf.gather(state_attrs, bond_graph_indices)
        edges_features_concated = tf.concat([edges_merge_atom_features, state_attrs_repeat, edges_sph_features], axis=-1)

        return edges_features_concated
        
    
    def aggregate_nodes(self, inputs: Sequence):
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
        atom_features, edges_sph_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        # concat state_attrs with atom_features to get merged atom_merge_state_features
        state_attrs_repeat = tf.gather(state_attrs, atom_graph_indices)
        atom_merge_state_features = tf.concat([atom_features, state_attrs_repeat], axis=-1)
        
        # using bond_updated to update atom_features_neighbors
        # using num_bonds of bond feature to renew num_bonds of adjacent atom feature
        # bond feature with shape (bond_dim,), not a Matrix, multiply by a learnable weight
        # with shape (atom_dim,atom_dim,bond_dim), then bond feature transfer to shape (atom_dim,atom_dim)
        # the bond matrix with shape (atom_dim,atom_dim) can update atom_feature with shape (aotm_dim,)
        # so num_bonds of bond features need num_bonds of bond matrix, so a matrix with shape
        # (num_bonds,(atom_dim,atom_dim,bond_dim)) to transfer bond_features to shape (num_bonds,(atom_dim,atom_dim))
        # finally, apply this bond_matrix to adjacent atoms, get bond_features updated atom_features_neighbors
        edges_weights = tf.matmul(edges_sph_features, self.kernel) + self.bias
        edges_weights = tf.reshape(edges_weights, (-1, self.atom_dim + self.state_dim, self.atom_dim + self.state_dim))
        atom_features_neighbors = tf.gather(atom_merge_state_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(edges_weights, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        
        # using conbination of tf.operation realizes multiplicationf between adjacent matrix and atom features
        # first tf.gather end features using end atom index pair_indices[:,1] to atom_features_neighbors
        # then using bond matrix updates atom_features_neighbors, get transformed_features
        # finally tf.segment_sum calculates sum of updated neighbors feature by start atom index pair_indices[:,0]
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:,0])

        return atom_features_aggregated
        
    
    def concat_states(self, inputs: Sequence):
        """
        Parameters
        ----------
        inputs : Sequence
            DESCRIPTION.

        Returns
        -------
        state_attrs_aggregated : TYPE
            DESCRIPTION.

        """
        atom_features, edges_sph_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # concat state_attrs with bond_updated and atom_features_aggregated
        edges_features_sum = tf.math.segment_sum(edges_sph_features, bond_graph_indices)
        atom_features_sum = tf.math.segment_sum(atom_features, atom_graph_indices)
        state_attrs_concated = tf.concat([atom_features_sum, edges_features_sum, state_attrs], axis=-1)

        return state_attrs_concated


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
        atom_features, edges_sph_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs
        
        atom_features_updated = atom_features
        edges_features_updated =  edges_sph_features
        state_attrs_updated = state_attrs

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # concate and update bond_features
            edges_features_concated = self.concat_edges(
                            [atom_features_updated, edges_features_updated,
                            state_attrs_updated, pair_indices, 
                            atom_graph_indices, bond_graph_indices]
                            )
            
            # Update edge_features via a step of GRU
            edges_features_updated, _ = self.update_edges(edges_features_concated, edges_features_updated)
            
            # Aggregate atom_features from neighbors
            atom_features_aggregated = self.aggregate_nodes(
                                    [atom_features_updated, edges_features_updated,
                                    state_attrs_updated, pair_indices,
                                    atom_graph_indices, bond_graph_indices]
                                    )

            # Update aggregated atom_features via a step of GRU
            atom_features_updated, _ = self.update_nodes(atom_features_aggregated, atom_features_updated)

            # update state_attrs
            state_attrs_concated = self.concat_states(
                                [atom_features_updated, edges_features_updated,
                                 state_attrs_updated, pair_indices,
                                 atom_graph_indices, bond_graph_indices]
                                )

            state_attrs_updated, _ = self.update_states(state_attrs_concated, state_attrs_updated)
            
        return [atom_features_updated, edges_features_updated, state_attrs_updated]
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config
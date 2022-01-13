# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 11:52:10 2022

@author: huzongxiang
"""

from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints
from crysnet.activations import shifted_softplus


class CrystalGraphConvolution(layers.Layer):
    """
    The CGCNN graph implementation as described in the paper
    Xie et al. PHYSICAL REVIEW LETTERS 120, 145301 (2018)
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
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_s',
            )
            self.bias_s = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_s',
            )

            self.kernel_g = self.add_weight(
                shape=(self.atom_dim * 2 + self.edge_dim, 
                       self.atom_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='nodes_kernel_g',
            )
            self.bias_g = self.add_weight(
                shape=(self.atom_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='nodes_bias_g',
            )
        
        self.softplus = shifted_softplus

        self.built = True
        
    
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
        atom_features_gather = tf.gather(atom_features, pair_indices)
        atom_merge_features = tf.concat([atom_features_gather[:,0], atom_features_gather[:,1], edges_sph_features], axis=-1)

        transformed_features_s = tf.matmul(atom_merge_features, self.kernel_s) + self.bias_s
        transformed_features_g = tf.matmul(atom_merge_features, self.kernel_g) + self.bias_g
        
        transformed_features = tf.sigmoid(transformed_features_s) * tf.nn.softplus(transformed_features_g)
        atom_features_aggregated = tf.math.segment_sum(transformed_features, pair_indices[:,0])

        atom_features_updated = atom_features + atom_features_aggregated
        atom_features_updated = tf.nn.softplus(atom_features_updated)

        return atom_features_updated


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
        
        for i in range(self.steps):
            atom_features_updated = self.aggregate_nodes([atom_features_updated, edges_sph_features,
                                                            state_attrs, pair_indices,
                                                            atom_graph_indices, bond_graph_indices])
            
        return atom_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"steps": self.steps})
        return config
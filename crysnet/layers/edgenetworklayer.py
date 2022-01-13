# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:33:33 2021

@author: hzx
"""

from typing import Sequence
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints
from crysnet.utils.spherical_harmonics import evaluate_spherical_harmonics


class SphericalBasisLayer(layers.Layer):
    
    def __init__(self, edge_dim=64,
                 num_spherical=6,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__( **kwargs)
        self.edge_dim = edge_dim
        self.num_spherical = num_spherical
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
    

    def build(self, input_shape):
        dense_dim = self.num_spherical**2

        with tf.name_scope("edge_aggregate"):
            self.kernel = self.add_weight(
                shape=(dense_dim, 
                    self.edge_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='sph_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.edge_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='sph_bias',
            )
        
        self.built = True
        

    def spherical_coordiates(self, local_env, pair_indices):
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        List
            return local spherical coordinate parameters theta and phi for every edges and their index.

        """
        local_env = tf.reshape(local_env, shape=(-1, 2, 3))
        sends, recives = pair_indices[:,0], pair_indices[:,1]
        edges_neighbor = tf.where(tf.expand_dims(sends,-1)==recives)
        polar = tf.gather(local_env, edges_neighbor[:,0])[:,0]
        edges = tf.gather(local_env, edges_neighbor[:,1])[:,0]
        vetical = tf.gather(local_env, edges_neighbor[:,0])[:,1]
        
        inner_product_theta = tf.reduce_sum(polar * edges, axis=-1)
        cross_product_theta = tf.linalg.cross(polar, edges)
        cross_product_theta = tf.norm(cross_product_theta, axis=-1)
        theta = tf.math.atan2(cross_product_theta, inner_product_theta)

        edges_projected = edges * tf.expand_dims(tf.cos(theta), axis=-1)  
        inner_product_phi = tf.reduce_sum(vetical * edges_projected, axis=-1)
        cross_product_phi = tf.linalg.cross(polar, edges_projected)
        cross_product_phi = tf.norm(cross_product_phi, axis=-1)
        phi = tf.math.atan2(cross_product_phi, inner_product_phi)
        
        phi_theta = tf.concat([tf.expand_dims(phi, -1), tf.expand_dims(theta, -1)], -1)
        
        return phi_theta, edges_neighbor


class GlobalSphericalBasisLayer(layers.Layer):
    
    def __init__(self, edge_dim=64,
                 num_spherical=6,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__( **kwargs)
        self.edge_dim = edge_dim
        self.num_spherical = num_spherical
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
    

    def build(self, input_shape):
        dense_dim = self.num_spherical**2

        with tf.name_scope("edge_aggregate"):
            self.kernel = self.add_weight(
                shape=(dense_dim, 
                    self.edge_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='sph_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.edge_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='sph_bias',
            )
        
        self.built = True
        

    def spherical_coordiates(self, local_env, pair_indices):
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        List
            return local spherical coordinate parameters theta and phi for every edges and their index.

        """
        local_env = tf.reshape(local_env, shape=(-1, 2, 3))
        polar = local_env[:,0]
        edges = local_env[:,1]
        
        inner_product_theta = tf.reduce_sum(polar * edges, axis=-1)
        cross_product_theta = tf.linalg.cross(polar, edges)
        cross_product_theta = tf.norm(cross_product_theta, axis=-1)
        theta = tf.math.atan2(cross_product_theta, inner_product_theta)

        edges_projected = edges * tf.expand_dims(tf.cos(theta), axis=-1)  
        inner_product_phi = tf.reduce_sum(vetical * edges_projected, axis=-1)
        cross_product_phi = tf.linalg.cross(polar, edges_projected)
        cross_product_phi = tf.norm(cross_product_phi, axis=-1)
        phi = tf.math.atan2(cross_product_phi, inner_product_phi)
        
        phi_theta = tf.concat([tf.expand_dims(phi, -1), tf.expand_dims(theta, -1)], -1)
        
        return phi_theta


    def sph_harm_func(self, phi_theta):
        phi = phi_theta[:, 0]
        theta = phi_theta[:, 1]
        Ylm = tf.stack([
                evaluate_spherical_harmonics(l, m, theta, phi) for l in range(self.num_spherical) for m in range(-l, l + 1)])
        sph_harm_features = tf.transpose(Ylm, perm=(1, 0))
        return sph_harm_features

        
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : theta and phi of edges in local coordinates.

        Returns
        -------
        Ylm : spherical harmonics for local coordinates theta and phi.
        """
        local_env, pair_indices = inputs
        phi_theta, edges_neighbor = self.spherical_coordiates(local_env, pair_indices)
        # sph_harm_features = tf.cast(self.sph_harm_func(1,0,phi_theta[:,0],phi_theta[:,1]).real, dtype=tf.float32)
        sph_harm_features = self.sph_harm_func(phi_theta)
        sph_harm_features = tf.matmul(sph_harm_features, self.kernel) + self.bias
        return sph_harm_features, edges_neighbor


    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        config.update({"num_spherical", self.num_spherical})
        return config


    def sph_harm_func(self, phi_theta):
        phi = phi_theta[:, 0]
        theta = phi_theta[:, 1]
        Ylm = tf.stack([
                evaluate_spherical_harmonics(l, m, theta, phi) for l in range(self.num_spherical) for m in range(-l, l + 1)])
        sph_harm_features = tf.transpose(Ylm, perm=(1, 0))
        return sph_harm_features

        
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : theta and phi of edges in local coordinates.

        Returns
        -------
        Ylm : spherical harmonics for local coordinates theta and phi.
        """
        local_env, pair_indices = inputs
        phi_theta, edges_neighbor = self.spherical_coordiates(local_env, pair_indices)
        # sph_harm_features = tf.cast(self.sph_harm_func(1,0,phi_theta[:,0],phi_theta[:,1]).real, dtype=tf.float32)
        sph_harm_features = self.sph_harm_func(phi_theta)
        sph_harm_features = tf.matmul(sph_harm_features, self.kernel) + self.bias
        return sph_harm_features, edges_neighbor


    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        config.update({"num_spherical", self.num_spherical})
        return config


class AzimuthLayer(layers.Layer):
    
    def __init__(self, units=64,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=False,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__( **kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
    
    
    def build(self, input_shape):
        if self.units is not None:
            dense_dim = self.units
        
        with tf.name_scope("edge_aggregate"):
            self.kernel = self.add_weight(
                shape=(2, dense_dim),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='azimuth_kernel',
            )

            self.bias = self.add_weight(
                shape=(dense_dim,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='azimuth_bias',
            )
        
        self.built = True
        

    def spherical_coordiates(self, local_env, pair_indices):
        """
        Parameters
        ----------
        graph : Dict
            DESCRIPTION.

        Returns
        -------
        List
            return local spherical coordinate parameters theta and phi for every edges and their index.

        """
        local_env = tf.reshape(local_env, shape=(-1, 2, 3))
        sends, recives = pair_indices[:,0], pair_indices[:,1]
        edges_neighbor = tf.where(tf.expand_dims(sends, -1)==recives)
        polar = tf.gather(local_env, edges_neighbor[:,0])[:,0]
        edges = tf.gather(local_env, edges_neighbor[:,1])[:,0]
        vetical = tf.gather(local_env, edges_neighbor[:,0])[:,1]
        
        inner_product_theta = tf.reduce_sum(polar * edges, axis=-1)
        cross_product_theta = tf.linalg.cross(polar, edges)
        cross_product_theta = tf.norm(cross_product_theta, axis=-1)
        theta = tf.math.atan2(cross_product_theta, inner_product_theta)

        edges_projected = edges*tf.expand_dims(tf.cos(theta), axis=-1)  
        inner_product_phi = tf.reduce_sum(vetical*edges_projected, axis=-1)
        cross_product_phi = tf.linalg.cross(polar, edges_projected)
        cross_product_phi = tf.norm(cross_product_phi, axis=-1)
        phi = tf.math.atan2(cross_product_phi, inner_product_phi)
        
        phi_theta = tf.concat([tf.expand_dims(phi, -1), tf.expand_dims(theta, -1)], -1)

        return phi_theta, edges_neighbor

        
    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : theta and phi of edges in local coordinates.

        Returns
        -------
        Ylm : spherical harmonics for local coordinates theta and phi.
        """
        local_env, pair_indices = inputs
        phi_theta, edges_neighbor = self.spherical_coordiates(local_env, pair_indices)

        azimuth_features = tf.matmul(phi_theta, self.kernel) + self.bias
        return azimuth_features, edges_neighbor
    
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class EdgeAggragate(layers.Layer):
    """
    This block is different from MessagePassomg block for aggregating and updating of Nodes,
    it is used to certain tasks that needs aggregating and updating edges of graph.
    """
    def __init__(self,
        units=64,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        activation="relu",
        **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        

    def build(self, input_shape):
        self.edge_dim = input_shape[0][-1]
        
        with tf.name_scope("edge_aggregate_sph"):
            self.kernel = self.add_weight(
                shape=(self.edge_dim, self.units),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='edge_sph_kernel',
            )

            self.bias = self.add_weight(
                shape=(self.units),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='edge_sph_bias',
            )
        
        self.built = True
    

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
        edges_sph_features, edges_neighbor = inputs
        
        # concat scalar bond_features with directional edges_sph_features 
        edges_features_neighbors = tf.gather(edges_sph_features, edges_neighbor[:,1])
        # edges_feature_concated = tf.concat([edges_features_gather, edges_sph_features], axis=-1)
        
        # aggregate edges_features
        edges_features_aggregated = tf.math.segment_sum(edges_features_neighbors, edges_neighbor[:,0])
        transformed_edges_features = tf.matmul(edges_features_aggregated, self.kernel) + self.bias

        # return edges_features_updated
        return transformed_edges_features 


    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class ConcatLayer(layers.Layer):
    def __init__(self, units=16,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None,
                kernel_constraint=None,
                use_bias=True,
                bias_initializer="zeros",
                bias_regularizer=None,
                bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint


    def build(self, input_shape):
        self.bond_dim = input_shape[0][-1]
        self.sph_dim = input_shape[1][-1]

        with tf.name_scope("edge_concat"):
            self.kernel = self.add_weight(
                shape=(self.bond_dim + self.sph_dim, self.units),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='concat_kernel',
            )

            self.bias = self.add_weight(
                shape=(self.units),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='concat_bias',
            )
        
        self.built = True


    def call(self, inputs):
        bond_features, edges_sph_features, edges_neighbor = inputs

        bond_features_gather = tf.gather(bond_features, edges_neighbor[:,1])
        edges_feature_concated = tf.concat([bond_features_gather, edges_sph_features], axis=-1)
        edges_features_aggregated = tf.math.segment_sum(edges_feature_concated, edges_neighbor[:,0])
        edges_features_aggregated = tf.matmul(edges_features_aggregated, self.kernel) + self.bias

        return edges_features_aggregated    


    def get_config(self):
        config = super().get_config()
        return config


class EdgeMessagePassing(layers.Layer):
    def __init__(self, units=64,
                 steps=3,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation="relu",
                 sph=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps
        self.sph = sph


    def build(self, input_shape):
        self.edge_dim = input_shape[0][-1]
        self.azimuth_step = AzimuthLayer()
        self.sph_step = SphericalBasisLayer()
        self.concat = ConcatLayer(units=self.edge_dim)
        self.message_step = EdgeAggragate(self.units)
        self.update_step = layers.GRUCell(self.edge_dim, name='update_sph_edge')
        self.layernorm = layers.LayerNormalization()
        self.built = True


    def call(self, inputs):
        bond_features, local_env, pair_indices = inputs

        if self.sph:
            edges_sph_features, edges_neighbor = self.sph_step([local_env, pair_indices])
        else:
            edges_sph_features, edges_neighbor = self.azimuth_step([local_env, pair_indices])

        edges_features_updated = self.concat([bond_features, edges_sph_features, edges_neighbor])
        # edges_features_updated = bond_features
        
        # Perform a number of steps of message passing
        # Aggregate atom_features from neighbors
        for i in range(self.steps):
            edges_features_aggregated = self.message_step(
                [edges_features_updated, edges_neighbor]
            )
            
            # Update aggregated atom_features via a step of GRU
            edges_features_updated, _ = self.update_step(
                edges_features_aggregated, edges_features_updated
            )
                
        # edge_features_updated = self.layernorm(edge_features_updated)
        
        return edges_features_updated
    

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
    
class EdgeGraphNetwork(layers.Layer):
    def __init__(self,
        steps,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        kernel_constraint=None,
        bias_initializer="zeros",
        bias_regularizer=None,
        bias_constraint=None,
        activation=None,
        sph=False,
        **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.sph = sph


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.edge_dim = input_shape[1][-1]
        self.state_dim = input_shape[2][-1]

        self.azimuth_step = AzimuthLayer()
        self.sph_step = SphericalBasisLayer(edge_dim=self.edge_dim)
        self.concat = ConcatLayer()
        self.update_nodes = layers.GRUCell(self.atom_dim, name='update_nodes')
        self.update_edges = layers.GRUCell(self.edge_dim, name='update_edges')
        self.update_states = layers.GRUCell(self.state_dim, name='update_states')
        
        with tf.name_scope("edges_aggregate"):
            # weight for updating atom_features by bond_features 
            self.kernel = self.add_weight(
                shape=(self.atom_dim, 
                       (self.edge_dim + self.state_dim)**2),
                trainable=True,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='edges_kernel',
            )
            self.bias = self.add_weight(
                shape=((self.edge_dim + self.state_dim)**2,),
                trainable=True,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='edges_bias',
            )
        
        self.built = True


    def concat_nodes(self, inputs: Sequence):
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
        edges_merge_sph_features = tf.math.segment_sum(edges_sph_features, pair_indices[:,0])
        
        # repeat state attributes by bond_graph_indices, then concatenate to bond_merge_atom_features
        state_attrs_repeat = tf.gather(state_attrs, atom_graph_indices)
        atoms_features_concated = tf.concat([atom_features, edges_merge_sph_features, state_attrs_repeat], axis=-1)

        return atoms_features_concated
        
    
    def aggregate_edges(self, inputs: Sequence):
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
        atom_features, edges_sph_features, state_attrs, edges_neighbor, atom_graph_indices, bond_graph_indices = inputs
        
        # concat state_attrs with edges_sph_features to get merged atom_merge_state_features
        state_attrs_repeat = tf.gather(state_attrs, bond_graph_indices)
        edge_merge_state_features = tf.concat([edges_sph_features, state_attrs_repeat], axis=-1)
        
        # using atom_updated to update edge_features_neighbors
        # using num_atoms of atom feature to renew num_atoms of adjacent edges feature
        # bond feature with shape (atom_dim,), not a Matrix, multiply by a learnable weight
        # with shape (edge_dim,edge_dim,atom_dim), then bond feature transfer to shape (edge_dim,edge_dim)
        # the bond matrix with shape (edge_dim,edge_dim) can update atom_feature with shape (edge_dim,)
        # so num_atoms of bond features need num_bonds of bond matrix, so a matrix with shape
        # (num_atoms,(edge_dim,edge_dim,atom_dim)) to transfer atom_features to shape (num_atoms,(edge_dim,edge_dim))
        # finally, apply this atom_matrix to adjacent edges, get atom_features updated edge_features_neighbors
        atom_features_gather = tf.gather(atom_features, tf.gather(pair_indices, edges_neighbor[:, 1])[:, 0])
        atom_weights = tf.matmul(atom_features_gather, self.kernel) + self.bias
        atom_weights = tf.reshape(atom_weights, (-1, self.edge_dim + self.state_dim, self.edge_dim + self.state_dim))
        edge_features_neighbors = tf.gather(edge_merge_state_features, edges_neighbor[:, 1])
        edge_features_neighbors = tf.expand_dims(edge_features_neighbors, axis=-1)
        transformed_features = tf.matmul(atom_weights, edge_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        
        # using conbination of tf.operation realizes multiplicationf between adjacent matrix and atom features
        # first tf.gather end features using end atom index pair_indices[:,1] to atom_features_neighbors
        # then using bond matrix updates atom_features_neighbors, get transformed_features
        # finally tf.segment_sum calculates sum of updated neighbors feature by start atom index pair_indices[:,0]
        edge_features_aggregated = tf.math.segment_sum(transformed_features, edges_neighbor[:,0])

        return edge_features_aggregated
        
    
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
        atom_features, bond_features, local_env, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        atom_features_updated = atom_features
        state_attrs_updated = state_attrs
        
        if self.sph:
            edges_sph_features, edges_neighbor = self.sph_step([local_env, pair_indices])
        else:
            edges_sph_features, edges_neighbor = self.azimuth_step([local_env, pair_indices])

        edges_features_updated = self.concat([bond_features, edges_sph_features, edges_neighbor])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # concate and update bond_features
            atoms_features_concated = self.concat_nodes(
                            [atom_features_updated, edges_features_updated,
                            state_attrs_updated, edges_neighbor, 
                            atom_graph_indices, bond_graph_indices]
                            )
            
            # Update edge_features via a step of GRU
            atoms_features_updated, _ = self.update_nodes(atoms_features_concated, atom_features_updated)
            
            # Aggregate atom_features from neighbors
            edges_features_aggregated = self.aggregate_edges(
                                    [atom_features_updated, edges_features_updated,
                                    state_attrs_updated, pair_indices,
                                    atom_graph_indices, bond_graph_indices]
                                    )

            # Update aggregated atom_features via a step of GRU
            edges_features_updated, _ = self.update_edges(edges_features_aggregated, edges_features_updated)

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
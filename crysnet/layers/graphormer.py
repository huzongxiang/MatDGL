# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:47:13 2021

@author: huzongxiang
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .graphnetworklayer import MessagePassing
from .crystalgraphlayer import GNConvolution


class ConvMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads=8,
                steps=1,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None,
                kernel_constraint=None,
                use_bias=True,
                bias_initializer="zeros",
                bias_regularizer=None,
                bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.steps = steps
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.supports_masking = True


    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.state_dim = input_shape[2][-1]

        self.query_conv = GNConvolution(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='query',
                                    )

        self.key_conv = GNConvolution(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='key',
                                    )

        self.value_conv = GNConvolution(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='value',
                                    )


    def call(self, inputs, mask=None):
        """
        Parameters
        ----------
        query : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.
        attn_bias : TYPE, optional
            DESCRIPTION. The default is None.
        attention_mask : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        attention_output : TYPE
            DESCRIPTION.

        """  
        atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # calculate Q=WqIn, K=WkIn, V=WvIn, In=Inputs
        atom_features_q = self.query_conv([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])
        atom_features_k = self.key_conv([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])
        atom_features_v = self.value_conv([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])

        atom_features_k = tf.transpose(atom_features_k, perm=(1, 0))

        # calculate Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        atom_features_scores = tf.matmul(atom_features_q, atom_features_k)/self.atom_dim**0.5
        atom_features_attention = tf.nn.softmax(atom_features_scores)
        atom_features_output = tf.matmul(atom_features_attention, atom_features_v)

        return atom_features_output
        

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"steps": self.steps})
        return config


class ConvGraphormerEncoder(layers.Layer):
    def __init__(self, num_heads=8,
                steps=1,
                embed_dim=16,
                edge_embed_dim=64,
                state_embed_dim=16,
                dense_dim=32,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None,
                kernel_constraint=None,
                use_bias=True,
                bias_initializer="zeros",
                bias_regularizer=None,
                bias_constraint=None,
                activation="relu",
                 **kwargs):
        super().__init__(**kwargs)

        self.attention = ConvMultiHeadAttention(num_heads,
                                                steps,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                use_bias=use_bias,
                                                bias_initializer=bias_initializer,
                                                bias_regularizer=bias_regularizer,
                                                bias_constraint=bias_constraint,
                                                )
        
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation, name='act_proj'), layers.Dense(embed_dim, name='proj'),]
        )
        
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

        self.supports_masking = True


    def call(self, inputs, mask=None):
        atom_features_output = self.attention(inputs, mask=mask)
        proj_input = self.layernorm_1(inputs[0] + atom_features_output)
        return self.layernorm_2(atom_features_output + self.dense_proj(proj_input))
    

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"embed_dim": self.embed_dim})
        config.update({"edge_embed_dim": self.edge_embed_dim})
        config.update({"state_embed_dim": self.state_embed_dim})
        config.update({"dense_dim": self.dense_dim})
        return config


class MpnnMultiHeadAttention(layers.Layer):
    def __init__(self, steps,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None,
                kernel_constraint=None,
                use_bias=True,
                bias_initializer="zeros",
                bias_regularizer=None,
                bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.supports_masking = True
        
        
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.state_dim = input_shape[2][-1]

        self.query_mpnn = MessagePassing(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='query',
                                    )

        self.key_mpnn = MessagePassing(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='key',
                                    )

        self.value_mpnn = MessagePassing(steps=self.steps,
                                    kernel_initializer=self.kernel_initializer,
                                    kernel_regularizer=self.kernel_regularizer,
                                    kernel_constraint=self.kernel_constraint ,
                                    bias_initializer=self.bias_initializer,
                                    bias_regularizer=self.bias_regularizer,
                                    recurrent_regularizer=None,
                                    bias_constraint=None,
                                    activation=None,
                                    name='value',
                                    )


    def call(self, inputs, mask=None):
        """
        Parameters
        ----------
        query : TYPE
            DESCRIPTION.
        key : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.
        attn_bias : TYPE, optional
            DESCRIPTION. The default is None.
        attention_mask : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        attention_output : TYPE
            DESCRIPTION.

        """  
        atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices = inputs

        # calculate Q=WqIn, K=WkIn, V=WvIn, In=Inputs
        atom_features_q, bond_features_q, state_attrs_q = self.query_mpnn([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])
        atom_features_k, bond_features_k, state_attrs_k = self.key_mpnn([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])
        atom_features_v, bond_features_v, state_attrs_v = self.value_mpnn([atom_features, bond_features, state_attrs, pair_indices, atom_graph_indices, bond_graph_indices])

        # calculate Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        atom_features_k = tf.transpose(atom_features_k, perm=(1, 0))
        bond_features_k = tf.transpose(bond_features_k, perm=(1, 0))
        state_attrs_k = tf.transpose(state_attrs_k, perm=(1, 0))

        atom_features_scores = tf.matmul(atom_features_q, atom_features_k)/self.atom_dim**0.5
        atom_features_attention = tf.nn.softmax(atom_features_scores)
        atom_features_output = tf.matmul(atom_features_attention, atom_features_v)

        bond_features_scores = tf.matmul(bond_features_q, bond_features_k)/self.bond_dim**0.5
        bond_features_attention = tf.nn.softmax(bond_features_scores)
        bond_features_output = tf.matmul(bond_features_attention, bond_features_v)

        state_attrs_scores = tf.matmul(state_attrs_q, state_attrs_k)/self.state_dim**0.5
        state_attrs_attention = tf.nn.softmax(state_attrs_scores)
        state_attrs_output = tf.matmul(state_attrs_attention, state_attrs_v)

        return atom_features_output, bond_features_output, state_attrs_output
        

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"steps": self.steps})
        return config


class GraphormerEncoder(layers.Layer):
    def __init__(self, num_heads=8,
                embed_dim=16,
                edge_embed_dim=64,
                state_embed_dim=16,
                dense_dim=32,
                kernel_initializer="glorot_uniform",
                kernel_regularizer=None,
                kernel_constraint=None,
                use_bias=True,
                bias_initializer="zeros",
                bias_regularizer=None,
                bias_constraint=None,
                activation="relu",
                 **kwargs):
        super().__init__(**kwargs)

        self.attention = MpnnMultiHeadAttention(num_heads,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                use_bias=use_bias,
                                                bias_initializer=bias_initializer,
                                                bias_regularizer=bias_regularizer,
                                                bias_constraint=bias_constraint,
                                                )
        
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation, name='act_proj'), layers.Dense(embed_dim, name='proj'),]
        )
        
        self.dense_proj_edge = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation, name='act_edge_proj'), layers.Dense(edge_embed_dim, name='edge_proj'),]
        )
        
        self.dense_proj_state = keras.Sequential(
            [layers.Dense(dense_dim, activation=activation, name='act_state_proj'), layers.Dense(state_embed_dim, name='state_proj'),]
        )
        
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        
        self.layernorm_edge_1 = layers.LayerNormalization()
        self.layernorm_edge_2 = layers.LayerNormalization()

        self.layernorm_state_1 = layers.LayerNormalization()
        self.layernorm_state_2 = layers.LayerNormalization()

        self.supports_masking = True


    def call(self, inputs, mask=None):
        atom_features_output, bond_features_output, state_attrs_output = self.attention(inputs, mask=mask)
        proj_input = self.layernorm_1(inputs[0] + atom_features_output)
        proj_input_edge = self.layernorm_edge_1(inputs[1] + bond_features_output)
        proj_input_state = self.layernorm_state_1(inputs[2] + state_attrs_output)
        return (self.layernorm_2(proj_input + self.dense_proj(proj_input)),
               self.layernorm_edge_2(proj_input_edge + self.dense_proj_edge(proj_input_edge)),
               self.layernorm_state_2(proj_input_state + self.dense_proj_state(proj_input_state))
        )
    

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"embed_dim": self.embed_dim})
        config.update({"edge_embed_dim": self.edge_embed_dim})
        config.update({"state_embed_dim": self.state_embed_dim})
        config.update({"dense_dim": self.dense_dim})
        return config
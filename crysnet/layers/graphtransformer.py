# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:47:13 2021

@author: huzongxiang
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class EdgesAugmentedLayer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.supports_masking = True
    
    def build(self, input_shape):
        self.feature_dim = input_shape[0][-1]
        self.pair_dim = input_shape[1][-1]

    
    def compute_mask(self, inputs, mask=None):
        return None


    def call(self, inputs, mask=None):
        edge_features_batch, pair_indices_batch = inputs
        
        feature_dim = self.feature_dim
        pair_dim = self.pair_dim
        
        batch_size = tf.shape(pair_indices_batch)[0]
        max_num_atoms = tf.reduce_max(tf.reshape(pair_indices_batch, (-1, 1))) + 1
        max_num_bonds = tf.shape(pair_indices_batch)[1] 
        
        initial_matrixs = tf.zeros((batch_size, max_num_atoms, max_num_atoms, feature_dim))
        batch_index = tf.cast(tf.expand_dims(tf.repeat(tf.range(batch_size), max_num_bonds), -1), dtype=tf.int64)
        indices = tf.cast(tf.reshape(pair_indices_batch, shape=(-1, pair_dim)), dtype=tf.int64)
        indices = tf.concat([batch_index, indices], -1)
        updates = tf.reshape(edge_features_batch, shape=(-1, feature_dim))
        
        augmented_matrixs = tf.tensor_scatter_nd_add(initial_matrixs, indices, updates)
        return augmented_matrixs


class GraphMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads,
                mask_value=0.,
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
        self.mask_value = mask_value
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.bias_regularizer = bias_regularizer
        self.bias_constraint = bias_constraint
        self.supports_masking = True
        
        
    def build(self, input_shape):
        self.query_dim = input_shape[0][-1]
        self.dense_dim = input_shape[0][-1]       
        self.edge_dim = input_shape[1][-1]
        
        self.out_dim = self.num_heads * self.dense_dim
        self.edge_out_dim = self.num_heads * self.edge_dim
        
        self.query_dense = layers.Dense(
            units=self.num_heads * self.query_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='query',
        )
        
        self.key_dense = layers.Dense(
            units=self.num_heads * self.query_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='key',
        )
        
        self.value_dense = layers.Dense(
            units=self.num_heads * self.dense_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='value',
        )

        self.dense_augmented = layers.Dense(
            units=self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='augmented',
        )
        
        self.dense_value = layers.Dense(
            units=self.num_heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='dense_value',
        )
        
        self.dense_output = layers.Dense(
            units=self.query_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='output',
        )

        self.dense_edge_output = layers.Dense(
            units=self.edge_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            name='edge_output',
        )

        self.activation_g = layers.Activation(activation='sigmoid')

        self.built = True


    def compute_mask(self, inputs, mask=None):
        # query mask
        query_mask = None
        scores_mask = None
        if mask is not None:
            query_mask = mask[0]
            q_mask = K.cast(query_mask, K.dtype(inputs[0]))
            q_mask = tf.expand_dims(q_mask, axis=-1)
            v_mask = K.permute_dimensions(q_mask, pattern=(0, 2, 1))
            scores_mask = tf.matmul(q_mask, v_mask)
            scores_mask = tf.expand_dims(scores_mask, axis=-1)
            scores_mask = K.any(K.not_equal(scores_mask, self.mask_value), axis=-1)

        return query_mask, scores_mask


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
        query, edges_augmented = inputs
        batch_size = tf.shape(query)[0]

        if mask is not None:
            if mask[0] is not None:
                boolean_mask = mask[0]

        # calculate Q=WqIn, K=WkIn, V=WvIn, In=Inputs
        Query = self.query_dense(query)
        Key = self.key_dense(query)
        Value = self.value_dense(query)

        # calculate Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        Query = tf.reshape(Query, shape=(batch_size, -1, self.num_heads, self.query_dim))
        Key = tf.reshape(Key, shape=(batch_size, -1, self.num_heads, self.query_dim))
        Value = tf.reshape(Value, shape=(batch_size, -1, self.num_heads, self.dense_dim))
        
        Query = tf.transpose(Query, perm=(0, 2, 1, 3))
        Key = tf.transpose(Key, perm=(0, 2, 3, 1))
        Value = tf.transpose(Value, perm=(0, 2, 1, 3))

        scores = tf.matmul(Query, Key)/self.query_dim**0.5
        attn_bias = self.dense_augmented(edges_augmented)
        attn_bias = tf.transpose(attn_bias, perm=(0, 3, 1, 2))
        scores = scores + attn_bias
        # scores = tf.matmul(scores, attn_bias)
        
        g_weights = self.dense_value(edges_augmented)
        g_weights = tf.transpose(g_weights, perm=(0, 3, 1 , 2))
        g_weights = self.activation_g(g_weights)

        # masking scores with -inf for softmax calculation
        if mask is not None:
            if mask[0] is not None:
                boolean_mask = tf.repeat(tf.expand_dims(boolean_mask, 1), self.num_heads, 1)
                q_mask = tf.expand_dims(K.cast(boolean_mask, K.dtype(query)), -1)
                v_mask = tf.transpose(q_mask, perm=(0, 1, 3, 2))
                scores_mask = tf.matmul(q_mask, v_mask)
                        
                masked_scores = scores * scores_mask - (1 - scores_mask) * 1e12
                g_weights = g_weights * scores_mask

                V = tf.einsum('ijkl,ijlm->ijkm', g_weights, Value)
        
                attention = tf.nn.softmax(masked_scores)
                attention = attention * scores_mask
                attention_output = tf.matmul(attention, V)
        else:
            masked_scores = scores
            V = tf.einsum('ijkl,ijlm->ijkm', g_weights, Value)
            attention = tf.nn.softmax(masked_scores)
            attention_output = tf.matmul(attention, V)

        attention_output = tf.reshape(attention_output, shape=(batch_size, -1, self.out_dim))
        attention_output = self.dense_output(attention_output)

        edges_output = tf.transpose(scores, perm=(0, 3, 2, 1))
        edges_output = self.dense_edge_output(edges_output)

        return attention_output, edges_output
        

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"mask_value": self.mask_value})
        return config

        
class GraphTransformerEncoder(layers.Layer):
    def __init__(self, num_heads=12,
                embed_dim=10,
                edge_embed_dim=16,
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

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.dense_dim = dense_dim

        self.attention = GraphMultiHeadAttention(num_heads,
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
        
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        
        self.layernorm_edge_1 = layers.LayerNormalization()
        self.layernorm_edge_2 = layers.LayerNormalization()
        
        # self.layernorm_1 = layers.BatchNormalization(trainable=True)
        # self.layernorm_2 = layers.BatchNormalization(trainable=True)
        
        # self.layernorm_edge_1 = layers.BatchNormalization(trainable=True)
        # self.layernorm_edge_2 = layers.BatchNormalization(trainable=True)

        self.supports_masking = True


    def call(self, inputs, mask=None):
        attention_output, edges_output = self.attention([inputs[0], inputs[1]], mask=mask)
        proj_input = self.layernorm_1(inputs[0] + attention_output)
        proj_input_edge = self.layernorm_edge_1(inputs[1] + edges_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input)),\
               self.layernorm_edge_2(proj_input_edge + self.dense_proj_edge(proj_input_edge))
    

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        config.update({"embed_dim": self.embed_dim})
        config.update({"edge_embed_dim": self.edge_embed_dim})
        config.update({"dense_dim": self.dense_dim})
        return config
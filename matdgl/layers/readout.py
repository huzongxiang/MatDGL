# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 20:15:01 2021

@author: huzongxiang
"""

from email.errors import InvalidMultipartContentTransferEncodingDefect
from pyexpat import features
from xml.sax.xmlreader import InputSource
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers, constraints


class Set2Set(layers.Layer):
    def __init__(self,
                 output_dim,
                 steps=1,
                 activation_lstm='tanh',
                 activation_recurrent='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.output_dim = output_dim
        self.steps = steps
        self.use_bias = use_bias
        self.activation_lstm = activations.get(activation_lstm)
        self.activation_recurrent = activations.get(activation_recurrent)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True
        

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        
        with tf.name_scope("set2set"):
            self.kernel_weight = self.add_weight(
                shape=(self.feature_dim, self.output_dim),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='set2set_kernel',
                )
            
            if self.use_bias:
                self.bias = self.add_weight(
                    shape=(self.output_dim),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    name='set2set_bias',
                    )
            else:
                self.bias = None

        
        self.recurrent_kernel = self.add_weight(shape=(self.output_dim * 2,
                                                self.output_dim * 4),
                                                initializer=self.recurrent_initializer,
                                                regularizer=self.recurrent_regularizer,
                                                constraint=self.recurrent_constraint,
                                                name='recurrent_kernel',
                                                )
        
        self.recurrent_a = self.add_weight(shape=(self.output_dim * 1,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.recurrent_regularizer,
                                            constraint=self.recurrent_constraint,
                                            name='recurrent_a',
                                            )

        self.recurrent_b = self.add_weight(shape=(self.output_dim * 1,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.recurrent_regularizer,
                                            constraint=self.recurrent_constraint,
                                            name='recurrent_b',
                                            )

        self.recurrent_c = self.add_weight(shape=(self.output_dim * 2,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.recurrent_regularizer,
                                            constraint=self.recurrent_constraint,
                                            name='recurrent_c',
                                            )

        self.built = True


    def call(self, inputs, mask=None, edge_mode=False, **kwargs):
        features = inputs
        batch_size = tf.shape(features)[0
                                        ]
        
        # if mask is not None:
        #     mask 
        
        if edge_mode:
            features = tf.reshape(features, (batch_size, -1, self.feature_dim))
        features = tf.matmul(features, self.kernel_weight) + self.bias
        
        # Set2Set embedding
        q_star = tf.reduce_sum(tf.zeros_like(features), axis=1, keepdims=True)  
        c = tf.zeros_like(q_star)  
        q_star = tf.concat([q_star, q_star], -1)  

        # set2set convert a list of vectors to a vector
        for i in range(self.steps):
            # q_t = LSTM(q∗_t−1)
            q, c = self._lstm(q_star, c)
            # print('q:',q.shape)
            # e_i,t = scalar(m_i,q_t)
            e = tf.einsum('ijk,ilk->ijl', features, q) 
            # print('e:', e.shape)
            # calculate a_i, t by softmax ei, t
            a = tf.nn.softmax(e, axis=1)  
            a = tf.tile(a, [1, 1, self.output_dim]) 
            # print(a.shape, features.shape)
            # calculated r_t by sum a_i, t and m_i
            r = tf.reduce_sum(tf.multiply(a, features), axis=1, keepdims=True)  
            q_star = tf.concat([q, r], -1)  

        return tf.reshape(q_star, [-1, self.output_dim * 2])  


    def _lstm(self, h, c):
        z = tf.matmul(h, self.recurrent_kernel)
        
        if self.use_bias:
           z = z + tf.concat([self.recurrent_a, self.recurrent_b, self.recurrent_c], -1)
        
        z_i = z[:, :, : self.output_dim]
        z_f = z[:, :, self.output_dim : self.output_dim * 2]
        z_o = z[:, :, self.output_dim * 2 : self.output_dim * 3]
        z_z = z[:, :, self.output_dim * 3 :]
        
        i = self.activation_recurrent(z_i)
        f = self.activation_recurrent(z_f)
        o = self.activation_recurrent(z_o)

        c_out = f * c + i * self.activation_lstm(z_z)
        h_out = o * self.activation_lstm(c_out)
        return h_out, c_out


    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim})
        config.update({"steps": self.steps})
        return config


class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, **kwargs):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_mask = mask[:, tf.newaxis, :] if mask is not None else None
        attention_output = self.attention(inputs, inputs, attention_mask=attention_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        return self.layernorm_2(proj_input + self.dense_proj(proj_input))


class LinearPredMasking(layers.Layer):
    def __init__(self,
                units=119,
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


    def build(self, inputs):

        self.dense = layers.Dense(
            units=self.units,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bias_regularizer,
            bias_constraint=self.bias_constraint,
            activation="softmax",
            name='dense',
            )
    

    def gather(self, features_list, masking_indices, masking_graph_indices):
        # using masking_graph_indice indexes the masking_indice in the batch
        # tf.stack index in form list [[0, 1], [0, 2], [1, 0], [1, 2], [1, 3] ...]
        # the "0" in [0, 1] indicates index in first dim, "1" indicates index in second dim
        # so tf.gather_nd first select 1st dim then select 2st dim through all indexes in the list
        indices = tf.stack([masking_graph_indices, masking_indices], axis=1)
        features = tf.gather_nd(features_list, indices)
        return features
        

    def call(self, inputs):
        features_list, masking_indices, masking_graph_indices = inputs
        features = self.gather(features_list, masking_indices, masking_graph_indices)
        features_batch = self.dense(features)

        return features_batch
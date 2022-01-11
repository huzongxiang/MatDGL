
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:47:13 2021

@author: huzongxiang
"""

import tensorflow as tf
from tensorflow.keras import layers


class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size


    def call(self, inputs):
        features, graph_indices = inputs

        # Obtain subgraphs
        features = tf.dynamic_partition(
            features, graph_indices, self.batch_size
        )

        # Pad and stack subgraphs
        num_features = [tf.shape(f)[0] for f in features]
        max_num = tf.reduce_max(num_features)
        features_padded = tf.stack(
            [
                tf.pad(f, [(0, max_num - n), (0, 0)])
                for f, n in zip(features, num_features)
            ],
            axis=0,
        )
        
        # Remove empty subgraphs (usually for last batch)
        nonempty_examples = tf.where(tf.reduce_sum(features_padded, (1, 2)) != 0)
        nonempty_examples = tf.squeeze(nonempty_examples, axis=-1)

        features_batch =  tf.gather(features_padded, nonempty_examples, axis=0)

        return features_batch
    
    
    def get_config(self):
        config = super().get_config()
        config.update({"batch": self.batch_size})
        return config
    

class PartitionPaddingPair(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size


    def call(self, inputs):
        features, graph_indices = inputs

        # Obtain subgraphs
        features = tf.dynamic_partition(
            features, graph_indices, self.batch_size
        )

        # Pad and stack subgraphs
        num_features = [tf.shape(f)[0] for f in features]
        max_num = tf.reduce_max(num_features)
        features_padded = tf.stack(
            [
                tf.pad(f, [(0, max_num - n), (0, 0)])
                for f, n in zip(features, num_features)
            ],
            axis=0,
        )
       
        
        # Remove empty subgraphs (usually for last batch)
        nonempty_examples = tf.unique(graph_indices)[0]

        features_batch =  tf.gather(features_padded, nonempty_examples, axis=0)

        return features_batch
        

    def get_config(self):
        config = super().get_config()
        config.update({"batch_size": self.batch_size})
        return config
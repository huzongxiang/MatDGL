# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:12:08 2021

@author: huzongxiang
"""

from tensorflow.keras.utils import to_categorical
from .crystalgraph import LabelledCrystalGraph, GraphBatchGeneratorSequence


class GraphGenerator:
    def __init__(self, dataset, data_size=None, batch_size=16, cutoff=3.0, mendeleev=False):
        self.dataset = dataset
        self.data_size = data_size
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.mendeleev = mendeleev
        self.multiclassification = None
        self.ntarget = 1

        self.train_generator, self.valid_generator, self.test_generator = self.generators()


    def generators(self):   
        structures = self.dataset.structures
        labels = self.dataset.labels
        task_type = self.dataset.task_type

        structures, labels = self.dataset.shuffle_set(structures, labels)

        if self.data_size:
            num = self.data_size
            structures_used = structures[:num]
            labels_used = labels[:num]
            labelledgraph = LabelledCrystalGraph(cutoff=self.cutoff, mendeleev=self.mendeleev)
            permutation = self.dataset.permute_indices(num)
        else:
            structures_used = structures
            labels_used = labels
            self.data_size = len(labels_used)
            labelledgraph = LabelledCrystalGraph(cutoff=self.cutoff, mendeleev=self.mendeleev)
            permutation = self.dataset.permute_indices(len(labels_used))

        print('prepare datasets, 70% for train, 20% for valid, 10% for test.')
        print('preparing train dataset...')
        x_train_, y_train = self.dataset.prepare_train_set(structures_used, labels_used, permutation)
        x_train = labelledgraph.inputs_from_strcutre_list(x_train_)

        print('preparing valid dataset...')
        x_valid_, y_valid = self.dataset.prepare_validate_set(structures_used, labels_used, permutation)
        x_valid = labelledgraph.inputs_from_strcutre_list(x_valid_)

        print('preparing test dataset...')
        x_test_, y_test = self.dataset.prepare_test_set(structures_used, labels_used, permutation)
        x_test = labelledgraph.inputs_from_strcutre_list(x_test_)

        if self.dataset.multiclassification:
            y_train = to_categorical(y_train)
            y_valid = to_categorical(y_valid)
            y_test = to_categorical(y_test)
            self.multiclassification = len(y_train[0])
        
        if self.dataset.regression:
            if isinstance(y_train[0], list):
                self.ntarget = len(y_train[0])

        train_data = GraphBatchGeneratorSequence(*x_train, y_train, task_type, batch_size=self.batch_size)
        valid_data = GraphBatchGeneratorSequence(*x_valid, y_valid, task_type, batch_size=self.batch_size)
        test_data = GraphBatchGeneratorSequence(*x_test, y_test, task_type, batch_size=self.batch_size)

        return train_data, valid_data, test_data


    def generator(self):   
        structures = self.dataset.structures
        labels = self.dataset.labels
        task_type = self.dataset.task_type

        if labels:
            structures, labels = self.dataset.shuffle_set(structures, labels)
        else:
            structures = self.dataset.shuffle_set(structures)

        if self.data_size:
            num = self.data_size
            structures_used = structures[:num]
            if labels:
                labels_used = labels[:num]
            labelledgraph = LabelledCrystalGraph(cutoff=self.cutoff, mendeleev=self.mendeleev)
        else:
            structures_used = structures
            if labels:
                labels_used = labels
            self.data_size = len(labels_used)
            labelledgraph = LabelledCrystalGraph(cutoff=self.cutoff, mendeleev=self.mendeleev)

        print('preparing dataset...')
        x_= self.dataset.prepare_x(structures_used)
        x = labelledgraph.inputs_from_strcutre_list(x_)

        if labels:
            y = self.dataset.prepare_y(labels_used)

            if self.dataset.multiclassification:
                y = to_categorical(y)
                self.multiclassification = len(y[0])

            if self.dataset.regression:
                if isinstance(y[0], list):
                    self.ntarget = len(y[0])

        data = GraphBatchGeneratorSequence(*x, y, task_type, batch_size=self.batch_size)

        return data
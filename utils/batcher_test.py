#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:26 2019

@author: littlelight
"""
import copy
import random
import numpy as np

from utils.data_handler_test import DataHandler
    
class Batcher:
    def __init__(self, data_handler):

        self.data_handler = data_handler

        data_splits = data_handler.get_data_splits()
        self.training_pairs = copy.deepcopy(data_splits[0])
        self.val_pairs = copy.deepcopy(data_splits[1])
        self.test_pairs = copy.deepcopy(data_splits[2])
        
        self.current_state = 0

    def get_training_size(self):
        return len(self.training_pairs)
        
    def get_training_batch(self, batch_size):
        """
        Get batch for training
        :param batch_size:
        :return:
        """
        if self.current_state == 0:
            random.shuffle(self.training_pairs)

        if (self.current_state + batch_size) > (len(self.training_pairs) + 1):
            self.current_state = 0
            return self.get_training_batch(batch_size)
        else:
            self.current_state += batch_size
            batch_pairs = self.training_pairs[self.current_state:(self.current_state + batch_size)]
            if len(batch_pairs) != batch_size:
                self.current_state = 0
                return self.get_training_batch(batch_size)
            return self.data_handler.slice_data(batch_pairs)

    def get_validation_batch(self, batch_size):
        """
        Validation batch for randomize, quantitative evaluation
        :param batch_size:
        :return:
        """
        if batch_size > len(self.val_pairs):
            return self.data_handler.slice_data(self.val_pairs)
        else:
            random.shuffle(self.val_pairs)
            return self.data_handler.slice_data(self.val_pairs[0:batch_size])

    def get_num_batches(self, batch_size):
        return int(float(len(self.training_pairs)) / float(batch_size))

    def get_num_training_subjects(self):
        return self.data_handler.get_num_training_subjects()
        
        

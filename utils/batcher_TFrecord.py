#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:26 2019

@author: littlelight
"""
import copy
import random
import numpy as np
import tensorflow as tf
from utils.data_handler_TFrecord import DataHandler

AUTOTUNE = tf.data.experimental.AUTOTUNE
    
class Batcher:
    def __init__(self, data_handler):

        self.data_handler = data_handler

        data_splits = data_handler.get_data_splits()
        self.training_shards = copy.deepcopy(data_splits[0])
        self.val_shards = copy.deepcopy(data_splits[1])
        self.test_shards = copy.deepcopy(data_splits[2])
        self.training_size = self._calculate_training_size()
        self.batch_size = data_handler.get_batch_size()
        
        ds_training = self.initialize_batch(self.training_shards)
        self.iterator_training = ds_training.make_one_shot_iterator()

        ds_val = self.initialize_batch(self.val_shards)
        self.iterator_val = ds_val.make_one_shot_iterator()

    def get_training_size(self):
        return self.training_size

    def initialize_batch(self, shards):
        """
        Get batch for training
        :param shards:
        :return:
        """
        files_ds = tf.data.Dataset.list_files(shards)

        # Disregard data order in favor of reading speed
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)

        # Read TFRecord files in an interleaved order
        ds = tf.data.TFRecordDataset(files_ds, num_parallel_reads=AUTOTUNE)
        # ds = tf.data.TFRecordDataset(files_ds_train, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)
        # Prepare batches
        ds = ds.repeat().batch(self.batch_size)

        # Parse a batch into a dataset of [audio, label] pairs
        ds = ds.map(lambda x: self._parse_batch(x))

        return ds

    def get_training_batch(self):
        audio_batch, blendshape_batch = self.iterator_training.get_next()
        return audio_batch, blendshape_batch

    def get_validation_batch(self):
        audio_batch, blendshape_batch = self.iterator_val.get_next()
        return audio_batch, blendshape_batch

    def get_num_batches(self, batch_size):
        return int(float(self.training_size) / float(batch_size))

    def get_num_training_subjects(self):
        return self.data_handler.get_num_training_subjects()

    def _calculate_training_size(self):
        training_size = 0
        for fn in self.training_shards:
            for record in tf.python_io.tf_record_iterator(fn):
                training_size += 1
        return training_size

    def _parse_batch(self, record_batch):
        window_size = self.data_handler.config['audio_window_size']
        num_features = self.data_handler.config['num_audio_features']
        num_frames = self.data_handler.config['num_consecutive_frames']
        num_blendshapes = self.data_handler.config['num_blendshapes']

        # Create a description of the features
        feature_description = {
            'audio': tf.io.FixedLenFeature([num_frames*num_features*window_size], tf.float32),
            'blendshape': tf.io.FixedLenFeature([num_blendshapes*num_frames], tf.float32),
        }

        # Parse the input `tf.Example` proto using the dictionary above
        example = tf.io.parse_example(record_batch, feature_description)
        example['audio'] = tf.reshape(example['audio'], [self.batch_size*num_frames, window_size, num_features])
        example['blendshape'] = tf.reshape(example['blendshape'], [self.batch_size*num_frames, num_blendshapes])
        print(example['audio'], example['blendshape'])
        return example['audio'], example['blendshape']

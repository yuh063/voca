#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:37:26 2019

@author: littlelight
"""

import csv
import os
import subprocess
import json
import numpy as np

from os.path import exists
from pydub import AudioSegment
from scipy.io.wavfile import read
#conda install scikit-learn
from sklearn.model_selection import train_test_split
from utils.audio_handler_test import AudioHandler

class DataHandler:
    def __init__(self, config): 
        # TODO need to add these parameters to config file
        folder_path = '/home/littlelight/voca/littlelights_animoji/'
        file_name = 'audio_jason_match.txt'
        
        self.config = config
        self.raw_audio = {}
        self.blendshape_dict = {}
        self.json_files = []
        self._process_data(folder_path, file_name)

    def get_data_splits(self):
        return self.train_keys, self.validate_keys, self.test_keys
        
    def _process_data(self, folder_path, file_name):
        with open(folder_path+file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_reader:
                if row:
                    [mp3_file, json_file] = row
                    self.json_files.append(json_file)
                    self._preprocess_audio(folder_path, mp3_file, json_file)
                    self.blendshape_dict[json_file] = self._read_json(folder_path, json_file)
        audio_handler = AudioHandler(self.config)
        
        # TODO need to implement this feature in the future
        '''
        if os.path.exists(processed_audio_path):
            self.processed_audio = pickle.load(open(processed_audio_path, 'rb'))
        else:
            self.processed_audio =  self._process_audio(self.raw_audio)
            if processed_audio_path != '':
                pickle.dump(self.processed_audio, open(processed_audio_path, 'wb'))
        '''
        
        self.processed_audio = audio_handler.process(self.raw_audio)
        if self.config['clear_unmatched_animation_pairs']:
            self._clear_unmatched_animation_pairs(self.json_files)
        self._init_data_splits()
            
    def _preprocess_audio(self, folder_path, mp3_file, json_file):
        wav_name = self._mp32wav(folder_path, mp3_file)
        downsample_name = 'downsampled_{}'.format(wav_name)
        wav_path = folder_path + wav_name
        downsample_path = folder_path + downsample_name
        self._downsample_file(wav_path, downsample_path)
        audio_arr = self._wav2arr(downsample_path)
        self.raw_audio[json_file] = audio_arr
    
    def _mp32wav(self, folder_path, mp3_name):
        mp3_path = folder_path+mp3_name
        wav_name = mp3_name[:-4] + '.wav'
        dst = folder_path + wav_name
        if not exists(dst):
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(dst, format="wav")
            # else:
            #    print(dst + ' already exists!')
        return wav_name

    def _downsample_file(self, in_path, out_path):
        try:
            # sr = 48000
            if not os.path.isfile(out_path):
                FNULL = open(os.devnull, 'w')
                completed = subprocess.call(['sox', in_path, '-r', str(self.config['sample_rate']), 
                                             '-b', '16', out_path], stdout=FNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            print(e)
            print('wrong format: {}'.format(in_path))

    def _wav2arr(self, wav_path):
        [sr, audio] = read(wav_path)
        if sr != self.config['sample_rate']:
            raise ValueError('Please downsample to sample rate 22000!')
        audio_arr = np.array(audio, dtype=float)
        return audio_arr
    
    def _read_json(self, folder_path, json_file):
        with open(folder_path+json_file) as JSON:
            json_dict = json.load(JSON)
            json_dict = self.byteify(json_dict)
            frames_dict = {}
            for idx, elm in enumerate(json_dict):
                frames_dict[idx] = elm
            return frames_dict
    
    def _clear_unmatched_animation_pairs(self, json_files):
        for key in json_files:
            num_data_frames, num_audio_frames = self._get_frames(key)
            
            # remove pairs which have time difference larger than threshold/frame_rate = 6/60 = 0.1s
            threshold = 6
            if abs(num_data_frames - num_audio_frames) > threshold:
                self.blendshape_dict.pop(key, None)
                self.processed_audio.pop(key, None)
                self.raw_audio.pop(key, None)
                
    
    def _init_data_splits(self):
        selected_files = self.raw_audio.keys()
        self.train_keys, test_and_validate_keys = train_test_split(selected_files, test_size=0.2)
        self.test_keys, self.validate_keys = train_test_split(test_and_validate_keys, test_size = 0.2)


    '''
    def _init_indices(self):
        def get_indices(keys):
            indices = []
            for key in keys:
                num_data_frames, num_audio_frames = self._get_frames(key)
                indices.append()
                
                try:
                    for i in range(min(num_data_frames, num_audio_frames)):
                        #indexed_frame = self.data2array_verts[subj][seq][i]
                        indices.append(indexed_frame)
                except KeyError:
                    print "Key error with subject: %s and sequence: %s" % (subj, seq)
       
            return indices

        self.training_indices = get_indices(self.training_subjects, self.training_sequences)
        self.validation_indices = get_indices(self.validation_subjects, self.validation_sequences)
        self.testing_indices = get_indices(self.testing_subjects, self.testing_sequences)

        self.training_idx2subj = {idx: self.training_subjects[idx] for idx in np.arange(len(self.training_subjects))}
        self.training_subj2idx = {self.training_idx2subj[idx]: idx for idx in self.training_idx2subj.keys()}
    '''
    
    def _get_frames(self, file_name):
        if (file_name not in self.blendshape_dict) or (file_name not in self.raw_audio):
            print 'missing file ', file_name
        num_data_frames = len(self.blendshape_dict[file_name])
        num_audio_frames = self.processed_audio[file_name].shape[0]
        return num_data_frames, num_audio_frames
    
    def byteify(self, input):
        if isinstance(input, dict):
            return {self.byteify(key): self.byteify(value) for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [self.byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input
      

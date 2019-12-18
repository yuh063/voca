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
import pickle
import pydub
import numpy as np

from os.path import exists
from pydub import AudioSegment
from scipy.io.wavfile import read
#conda install scikit-learn
from sklearn.model_selection import train_test_split
from utils.audio_handler_test import AudioHandler
from tqdm import tqdm

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.raw_audio = {}
        self.blendshape_dict = {}
        self.json_files = []
        self.file_frame_dict = {}
        self.processed_audio = {}

        folder_path = self.config['dataset_path']
        file_name = self.config['audio_json_pair_file_name']
        self._process_data(folder_path, file_name)

    def get_data_splits(self):
        return self.train_pairs, self.validate_pairs, self.test_pairs

    def slice_data(self, pairs):
        return self._slice_data_helper(pairs)

    def byteify(self, input):
        if isinstance(input, dict):
            return {self.byteify(key): self.byteify(value) for key, value in input.iteritems()}
        elif isinstance(input, list):
            return [self.byteify(element) for element in input]
        elif isinstance(input, unicode):
            return input.encode('utf-8')
        else:
            return input

    def _process_data(self, folder_path, file_name):
        if self._processed_data_exist(folder_path):
            print('load file_frame_dict')
            # pickle_in = open("{}file_frame_dict.pickle".format(folder_path), "rb")
            # self.file_frame_dict = pickle.load(pickle_in)
            with open("{}file_frame_dict.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            with open("{}file_frame_dict_1.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            with open("{}file_frame_dict_2.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            print('load blendshape_dict')
            # pickle_in = open("{}blendshape_dict.pickle".format(folder_path), "rb")
            # self.blendshape_dict = pickle.load(pickle_in)
            with open("{}blendshape_dict.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            with open("{}blendshape_dict_1.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            with open("{}blendshape_dict_2.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            print('load processed_audio')
            # pickle_in = open("{}processed_audio.pickle".format(folder_path), "rb")
            # self.processed_audio = pickle.load(pickle_in)
            with open("{}processed_audio.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
            with open("{}processed_audio_1.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
            with open("{}processed_audio_2.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
        else:
            blendshape_raw_dict = {}
            with open(folder_path+file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                print("load raw data")
                count = 0
                for row in tqdm(csv_reader):
                    if count <= 5000:
                        count += 1
                        continue
                    if row:
                        [mp3_file, json_file] = row
                        self.json_files.append(json_file)
                        try:
                            self._preprocess_audio(folder_path, mp3_file, json_file)
                            blendshape_raw_dict[json_file] = self._read_json(folder_path, json_file)
                        except pydub.exceptions.CouldntDecodeError:
                            self.json_files.remove(json_file)
                            print('{} is not a valid mp3 file.'.format(mp3_file))
                        except ValueError:
                            self.json_files.remove(json_file)
                            print('{} is not a valid mp3 file.'.format(mp3_file))
                        count += 1
                    if count > 7000:
                        break
            audio_handler = AudioHandler(self.config)
            print("build blendshape dictionary")
            self._reconstruct_blenshape_dict(blendshape_raw_dict)
            print("process audio")
            self.processed_audio = audio_handler.process(self.raw_audio)
            self._build_file_frame_dict()
            if self.config['clear_unmatched_animation_pairs']:
                self._clear_unmatched_animation_pairs(self.json_files)
            self._save_dicts(folder_path)
        if self._check_data_split(folder_path):
            with open("{}train_pairs.pickle".format(folder_path), "rb") as f:
                self.train_pairs = pickle.load(f)
            with open("{}test_pairs.pickle".format(folder_path), "rb") as f:
                self.test_pairs.update(pickle.load(f))
            with open("{}validation_pairs.pickle".format(folder_path), "rb") as f:
                self.validation_pairs.update(pickle.load(f))
        else:
            self._init_data_splits()
            self._save_pairs(folder_path)
        print("data split complete")

    def _processed_data_exist(self, folder_path):
        if exists("{}file_frame_dict.pickle".format(folder_path)) and exists("{}blendshape_dict.pickle".format(folder_path)) \
                and exists("{}processed_audio.pickle".format(folder_path)):
            return True
        else:
            return False

    def _reconstruct_blenshape_dict(self, blendshape_raw_dict):
        for key in tqdm(blendshape_raw_dict.keys()):
            frames = {}
            for frame in blendshape_raw_dict[key]:
                blendshapes = np.zeros(shape=(self.config['num_blendshapes']), dtype=float)
                for blendshape in blendshape_raw_dict[key][frame]['frame']:
                    blendshapes[int(blendshape)] = float(blendshape_raw_dict[key][frame]['frame'][blendshape])
                frames[frame] = blendshapes
            if len(blendshape_raw_dict[key]) < 30:
                self.raw_audio.pop(key, None)
                self.json_files.remove(key)
                continue
            end_time = float(blendshape_raw_dict[key][len(blendshape_raw_dict[key])-1]['time'])
            start_time = float(blendshape_raw_dict[key][0]['time'])
            num_frames = len(blendshape_raw_dict[key])
            if self._validate_fps(end_time, start_time, num_frames):
                self.blendshape_dict[key] = frames
            else:
                self.raw_audio.pop(key, None)
                self.json_files.remove(key)

    def _validate_fps(self, end_time, start_time, num_frames, fps=60, fps_threshold=2):
        average_time_per_frame = (end_time - start_time)/(num_frames - 1)
        estimate_fps = 1/average_time_per_frame
        if abs(fps - estimate_fps) > fps_threshold:
            return False
        else:
            return True

    def _build_file_frame_dict(self):
        for key in self.json_files:
            num_data_frames, num_audio_frames = self._get_frames(key)
            frame = min(num_data_frames, num_audio_frames)
            if frame:
                self.file_frame_dict[key] = frame

    def _preprocess_audio(self, folder_path, mp3_file, json_file):
        wav_name = self._mp32wav(folder_path, mp3_file)
        wav_path = folder_path + wav_name
        audio_arr = self._wav2arr(wav_path)
        self.raw_audio[json_file] = audio_arr

    def _mp32wav(self, folder_path, mp3_name):
        mp3_path = folder_path+mp3_name
        wav_name = mp3_name[:-4] + '.wav'
        dst = folder_path + wav_name
        if not exists(dst):
            sound = AudioSegment.from_mp3(mp3_path)
            sound.export(dst, format="wav")
        return wav_name

    def _wav2arr(self, wav_path):
        [sr, audio] = read(wav_path)
        if sr != self.config['sample_rate']:
            raise ValueError('Please provide file with sample rate {}}!'.format(self.comfig['sample_rate']))
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
            
            # remove pairs which have time difference longer than threshold/frame_rate = 6/60 = 0.1s
            threshold = 6
            if abs(num_data_frames - num_audio_frames) > threshold:
                self.blendshape_dict.pop(key, None)
                self.processed_audio.pop(key, None)
                self.raw_audio.pop(key, None)
                self.file_frame_dict.pop(key, None)

    def _save_dicts(self, folder_path):
        with open("{}processed_audio.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.processed_audio, pickle_out)
        with open("{}blendshape_dict.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.blendshape_dict, pickle_out)
        with open("{}file_frame_dict.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.file_frame_dict, pickle_out)

    def _check_data_split(self, folder_path):
        if exists("{}train_pairs.pickle".format(folder_path)) and exists("{}test_pairs.pickle".format(folder_path)) \
                and exists("validate_pairs.pickle".format(folder_path)):
            return True
        else:
            return False

    def _init_data_splits(self):
        selected_files = self.file_frame_dict.keys()
        self.train_keys, test_and_validate_keys = train_test_split(selected_files, test_size=0.2)
        self.test_keys, self.validate_keys = train_test_split(test_and_validate_keys, test_size = 0.5)
        self.train_pairs = self._get_file_frame_pair(self.train_keys)
        self.test_pairs = self._get_file_frame_pair(self.test_keys)
        self.validate_pairs = self._get_file_frame_pair(self.validate_keys)

    def _save_pairs(self, folder_path):
        with open("{}train_pairs.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.train_pairs, pickle_out)
        with open("{}test_pairs.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.train_pairs, pickle_out)
        with open("{}validate_pairs.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.train_pairs, pickle_out)
        with open('test_listfile.txt', 'w') as filehandle:
            for listitem in self.test_keys:
                filehandle.write('%s\n' % listitem)

    def _get_file_frame_pair(self, keys):
        file_frame_pair = []
        for key in keys:
            frame = self.file_frame_dict[key]
            for i in xrange(frame - 1):
                if i % 2:
                    file_frame_pair.append([key, i])
        return file_frame_pair

    def _get_frames(self, file_name):
        if (file_name not in self.blendshape_dict) or (file_name not in self.raw_audio):
            print 'missing file ', file_name
            return 0, 0
        num_data_frames = len(self.blendshape_dict[file_name])
        num_audio_frames = self.processed_audio[file_name].shape[0]
        return num_data_frames, num_audio_frames

    def _slice_data_helper(self, pairs):
        blendshapes = []
        processed_audio = []
        for pair in pairs:
            blendshapes.append(self.blendshape_dict[pair[0]][pair[1]])
            blendshapes.append(self.blendshape_dict[pair[0]][pair[1]]+1)
            processed_audio.append(self.processed_audio[pair[0]][pair[1], :, :])
            processed_audio.append(self.processed_audio[pair[0]][pair[1]+1, :, :])
        blendshapes = np.stack(blendshapes)
        processed_audio = np.stack(processed_audio)
        return processed_audio, blendshapes


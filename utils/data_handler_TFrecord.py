"""
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
"""

import csv
import json
import pickle
import pydub
import glob
import itertools
import numpy as np
import tensorflow as tf

from os.path import exists
from pydub import AudioSegment
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from utils.audio_handler_test import AudioHandler
from tqdm import tqdm


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.shard_size = 500
        self.TFRecord_file_number = 0
        self._clear_variables()

        folder_path = self.config['dataset_path']
        file_name = self.config['audio_json_pair_file_name']
        self._load_data(folder_path, file_name)

    def get_data_splits(self):
        return self.train_shards, self.validate_shards, self.test_shards

    def slice_data(self, pairs):
        return self._slice_data_helper(pairs)

    def get_batch_size(self):
        return self.config['batch_size']

    def byteify(self, input_instance):
        if isinstance(input_instance, dict):
            return {self.byteify(key): self.byteify(value) for key, value in input_instance.iteritems()}
        elif isinstance(input_instance, list):
            return [self.byteify(element) for element in input_instance]
        elif isinstance(input_instance, unicode):
            return input_instance.encode('utf-8')
        else:
            return input_instance

    def _load_data(self, folder_path, file_name):
        if self._check_data_split(folder_path):
            print("load stored data")
            with open("{}train_shards.pickle".format(folder_path), "rb") as f:
                self.train_shards = pickle.load(f)
            with open("{}test_shards.pickle".format(folder_path), "rb") as f:
                self.test_shards = pickle.load(f)
            with open("{}validate_shards.pickle".format(folder_path), "rb") as f:
                self.validate_shards = pickle.load(f)
        else:
            with open(folder_path+file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                print("load raw data")
                rows = []
                for row in csv_reader:
                    if row:
                        rows.append(row)
                        if len(rows) >= self.shard_size:
                            self._process_data(folder_path, rows)
                            self.TFRecord_file_number += 1
                            rows = []
                if len(rows) > 0:
                    self._process_data(folder_path, rows)
                self._init_data_splits(folder_path)
                self._save_pairs(folder_path)
                print("data split complete")

    def _process_data(self, folder_path, rows):
        blendshape_raw_dict = {}
        self._clear_variables()
        for row in rows:
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
        audio_handler = AudioHandler(self.config)
        print("process audio for shard {}".format(self.TFRecord_file_number))
        self.processed_audio = audio_handler.process(self.raw_audio)
        print("build blendshape dictionary for shard {}".format(self.TFRecord_file_number))
        self._reconstruct_blenshape_dict(blendshape_raw_dict)
        self._build_file_frame_dict()
        # print(self.processed_audio)
        # print(self.blendshape_dict)
        if self.config['clear_unmatched_animation_pairs']:
            self._clear_unmatched_animation_pairs(self.json_files)
        print(self.file_frame_dict.items())
        self._save_TFRecord_shards(folder_path)
        print('save TFRecord complete')

    def _clear_variables(self):
        self.raw_audio = {}
        self.processed_audio = {}
        self.blendshape_dict = {}
        self.json_files = []
        self.file_frame_dict = {}

    @staticmethod
    def _processed_data_exist(folder_path):
        if exists("{}file_frame_dict.pickle".format(folder_path)) and exists("{}blendshape_dict.pickle".format(folder_path)) \
                and exists("{}processed_audio.pickle".format(folder_path)):
            return True
        else:
            return False

    def _reconstruct_blenshape_dict(self, blendshape_raw_dict):
        for key in tqdm(blendshape_raw_dict.keys()):
            frames = {}
            if len(blendshape_raw_dict[key]) < self.config['frame_rate'] / 2.0:
                self.raw_audio.pop(key, None)
                self.json_files.remove(key)
                continue
            estimate_fps = self._estimate_fps(blendshape_raw_dict[key])
            if not self._valid_fps(estimate_fps, self.config['frame_rate']):
                self.raw_audio.pop(key, None)
                self.json_files.remove(key)
                continue
            if self._valid_fps(estimate_fps, self.config['frame_rate']) == self.config['frame_rate']:
                for frame in blendshape_raw_dict[key]:
                    blendshapes = np.zeros(shape=(self.config['num_blendshapes']), dtype=float)
                    for blendshape in blendshape_raw_dict[key][frame]['frame']:
                        blendshapes[int(blendshape)] = float(blendshape_raw_dict[key][frame]['frame'][blendshape])
                    frames[frame] = blendshapes
            else:
                for frame in blendshape_raw_dict[key]:
                    if not frame % 2:
                        blendshapes = np.zeros(shape=(self.config['num_blendshapes']), dtype=float)
                        for blendshape in blendshape_raw_dict[key][frame]['frame']:
                            blendshapes[int(blendshape)] = float(blendshape_raw_dict[key][frame]['frame'][blendshape])
                        frames[frame/2] = blendshapes
            self.blendshape_dict[key] = frames

    @staticmethod
    def _estimate_fps(blendshape_raw_item):
        end_time = float(blendshape_raw_item[len(blendshape_raw_item) - 1]['time'])
        start_time = float(blendshape_raw_item[0]['time'])
        num_frames = len(blendshape_raw_item)
        average_time_per_frame = (end_time - start_time) / (num_frames - 1)
        estimate_fps = 1 / average_time_per_frame
        return estimate_fps

    @staticmethod
    def _valid_fps(estimate_fps, fps, fps_threshold=2):
        # print(estimate_fps)
        if abs(fps - estimate_fps) < fps_threshold:  # fps can only be 30 or 60
            return fps
        elif abs(60 - estimate_fps) < fps_threshold:
            return 60
        else:
            return 0

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
            raise ValueError('Please provide file with sample rate {}}!'.format(self.config['sample_rate']))
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
            
            # remove pairs which have time difference longer than threshold/frame_rate = 3/30 = 0.1s
            threshold = 3
            if abs(num_data_frames - num_audio_frames) > threshold:
                self.blendshape_dict.pop(key, None)
                self.processed_audio.pop(key, None)
                self.raw_audio.pop(key, None)
                self.file_frame_dict.pop(key, None)

    def _save_TFRecord_shards(self, folder_path):
        tfrecords_filename = '{}littlelights_ai_{}_{}.tfrecords'.format(folder_path, self.config['num_consecutive_frames'], self.TFRecord_file_number)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)

        for file in self.file_frame_dict.keys():
            frame = self.file_frame_dict[file]
            for i in xrange(frame - self.config['num_consecutive_frames']):
                if not i % self.config['num_consecutive_frames']:
                    audio = self.processed_audio[file][i:i+self.config['num_consecutive_frames'], :, :].flatten().tolist()
                    blendshape = list(itertools.chain(*(self.blendshape_dict[file][k] for k in range(i, i+self.config['num_consecutive_frames']))))
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'file': self._bytes_feature(file),
                        'frame': self._int64_feature([frame, i, i+self.config['num_consecutive_frames']]),
                        'audio': self._float_feature(audio),
                        'blendshape': self._float_feature(blendshape)}))
                    writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _check_data_split(self, folder_path):
        if exists("{}train_shards.pickle".format(folder_path)) and exists("{}test_shards.pickle".format(folder_path)) \
                and exists("{}validate_shards.pickle".format(folder_path)):
            return True
        else:
            return False

    def _init_data_splits(self, folder_path):
        selected_files = glob.glob(folder_path + '*.tfrecords')
        self.train_shards, test_and_validate_shards = train_test_split(selected_files, test_size=0.2)
        self.test_shards, self.validate_shards = train_test_split(test_and_validate_shards, test_size = 0.5)

    def _save_pairs(self, folder_path):
        with open("{}train_shards.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.train_shards, pickle_out)
        with open("{}test_shards.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.test_shards, pickle_out)
        with open("{}validate_shards.pickle".format(folder_path), "wb") as pickle_out:
            pickle.dump(self.validate_shards, pickle_out)
        with open('{}test_shards_filename.txt'.format(folder_path), 'w') as filehandle:
            for shard in self.test_shards:
                filehandle.write('%s\n' % shard)

    def _get_frames(self, file_name):
        if (file_name not in self.blendshape_dict) or (file_name not in self.processed_audio):
            print 'missing file ', file_name
            return 0, 0
        num_data_frames = len(self.blendshape_dict[file_name])
        num_audio_frames = self.processed_audio[file_name].shape[0]
        return num_data_frames, num_audio_frames

    def _slice_data_helper(self, pairs):
        blendshapes = []
        processed_audio = []
        for pair in pairs:
            for i in range(self.config['num_consecutive_frames']):
                blendshapes.append(self.blendshape_dict[pair[0]][pair[1]+i])
                processed_audio.append(self.processed_audio[pair[0]][pair[1]+i, :, :])
        blendshapes = np.stack(blendshapes)
        processed_audio = np.stack(processed_audio)
        return processed_audio, blendshapes


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
import numpy as np

from os.path import exists
from pydub import AudioSegment
from scipy.io.wavfile import read
from sklearn.model_selection import train_test_split
from utils.audio_handler_test import AudioHandler
from tqdm import tqdm


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.raw_audio = {}
        self.processed_audio = {}
        self.blendshape_dict = {}
        self.json_files = []
        self.file_frame_dict = {}

        folder_path = self.config['dataset_path']
        file_name = self.config['audio_json_pair_file_name']
        self._process_data(folder_path, file_name)

    def get_data_splits(self):
        return self.train_pairs, self.validate_pairs, self.test_pairs

    def slice_data(self, pairs):
        return self._slice_data_helper(pairs)

    def byteify(self, input_instance):
        if isinstance(input_instance, dict):
            return {self.byteify(key): self.byteify(value) for key, value in input_instance.iteritems()}
        elif isinstance(input_instance, list):
            return [self.byteify(element) for element in input_instance]
        elif isinstance(input_instance, unicode):
            return input_instance.encode('utf-8')
        else:
            return input_instance

    def _process_data(self, folder_path, file_name):
        if self._processed_data_exist(folder_path):
            # TODO write a function to wrap
            pickle_in = open("{}file_frame_dict.pickle".format(folder_path), "rb")
            self.file_frame_dict = pickle.load(pickle_in)
            pickle_in = open("{}blendshape_dict.pickle".format(folder_path), "rb")
            self.blendshape_dict = pickle.load(pickle_in)
            pickle_in = open("{}processed_audio.pickle".format(folder_path), "rb")
            self.processed_audio = pickle.load(pickle_in)

            '''
            print('load file_frame_dict')
            with open("{}file_frame_dict.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            with open("{}file_frame_dict_1.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            with open("{}file_frame_dict_2.pickle".format(folder_path), "rb") as f:
                self.file_frame_dict.update(pickle.load(f))
            print('load blendshape_dict')
            with open("{}blendshape_dict.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            with open("{}blendshape_dict_1.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            with open("{}blendshape_dict_2.pickle".format(folder_path), "rb") as f:
                self.blendshape_dict.update(pickle.load(f))
            print('load processed_audio')
            with open("{}processed_audio.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
            with open("{}processed_audio_1.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
            with open("{}processed_audio_2.pickle".format(folder_path), "rb") as f:
                self.processed_audio.update(pickle.load(f))
            '''

        else:
            blendshape_raw_dict = {}
            with open(folder_path+file_name) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter='|')
                print("load raw data")
                for row in tqdm(csv_reader):
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

            audio_handler = AudioHandler(self.config)
            print("build blendshape dictionary")
            self._reconstruct_blenshape_dict(blendshape_raw_dict)
            print("process audio")
            self.processed_audio = audio_handler.process(self.raw_audio)
            self._build_file_frame_dict()
            print(self.processed_audio)
            print(self.blendshape_dict)
            print("debug")
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
            end_time = float(blendshape_raw_dict[key][len(blendshape_raw_dict[key])-1]['time'])
            start_time = float(blendshape_raw_dict[key][0]['time'])
            num_frames = len(blendshape_raw_dict[key])
            if len(blendshape_raw_dict[key]) < 30 or not self._valid_fps(end_time, start_time, num_frames):
                self.raw_audio.pop(key, None)
                self.json_files.remove(key)
                continue
            for frame in blendshape_raw_dict[key]:
                blendshapes = np.zeros(shape=(self.config['num_blendshapes']), dtype=float)
                for blendshape in blendshape_raw_dict[key][frame]['frame']:
                    blendshapes[int(blendshape)] = float(blendshape_raw_dict[key][frame]['frame'][blendshape])
                frames[frame] = blendshapes
            self.blendshape_dict[key] = frames

    def _valid_fps(self, end_time, start_time, num_frames, fps=60, fps_threshold=2):
        average_time_per_frame = (end_time - start_time)/(num_frames - 1)
        estimate_fps = 1/average_time_per_frame
        print(estimate_fps)
        if abs(fps - estimate_fps) > fps_threshold:
            return False
        else:
            return True

    def _build_file_frame_dict(self):
        for key in self.json_files:
            num_data_frames, num_audio_frames = self._get_frames(key)
            print(num_data_frames, num_audio_frames, key)
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
            
            # remove pairs which have time difference longer than threshold/frame_rate = 6/60 = 0.1s
            threshold = 6
            if abs(num_data_frames - num_audio_frames) > threshold:
                print(num_data_frames, num_audio_frames)
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
        with open('{}test_listfile.txt'.format(folder_path), 'w') as filehandle:
            for listitem in self.test_keys:
                filehandle.write('%s\n' % listitem)

    def _get_file_frame_pair(self, keys):
        file_frame_pair = []
        for key in keys:
            frame = self.file_frame_dict[key]
            for i in xrange(frame - self.config['num_consecutive_frames']):
                if not i % self.config['num_consecutive_frames']:
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
            for i in range(self.config['num_consecutive_frames']):
                blendshapes.append(self.blendshape_dict[pair[0]][pair[1]+i])
                processed_audio.append(self.processed_audio[pair[0]][pair[1]+i, :, :])
        blendshapes = np.stack(blendshapes)
        processed_audio = np.stack(processed_audio)
        return processed_audio, blendshapes


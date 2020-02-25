'''
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
'''


import re
import os
import json
import time
import numpy as np
import tensorflow as tf
import scipy

from scipy.io import wavfile
from scipy.io.wavfile import read
from scipy.ndimage import uniform_filter1d
from pydub import AudioSegment
from glob import glob
from audio_handler_test import AudioHandler
from hparams import load_hparams


def process_audio(config, audio):
    audio_arr = np.array(audio, dtype=float)
    tmp_audio = {'tmp': audio_arr}

    audio_handler = AudioHandler(config)

    return audio_handler.process(tmp_audio)['tmp']


def output_sequences(sequence_blendshapes, out_path, audio_fname, config):
    output = []
    current_time = time.time()
    trans_list = ["1.000", "0.000", "0.000", "0.000", "0.000", "1.000", "0.000", "0.000", "0.000", "0.000", "1.000",
                  "0.000", "0.000", "0.000", "0.000", "1.000"]
    sequence_blendshapes = uniform_filter1d(sequence_blendshapes, size=5, axis=0)  # smoothen blendshapes
    weight = np.ones(51)
    weight[42] = 1.1
    weight[31] = 1.3
    weight[49] = 1.1
    sequence_blendshapes = sequence_blendshapes * weight
    sequence_blendshapes[:, 42] -= 0.005
    sequence_blendshapes[:, 31] -= 0.01
    sequence_blendshapes[:, 49] -= 0.005
    for i in range(sequence_blendshapes.shape[0]):
        current_time += 1.0/30
        frame_dict = {}
        tmp_dict = {}
        for j in range(sequence_blendshapes.shape[1]):
            scale = sequence_blendshapes[i, j]
            if scale > 1:
                scale = 1
            elif scale < 0:
                scale = 0
            tmp_dict[str(j)] = "{:.3f}".format(scale)
        frame_dict["frame"] = tmp_dict
        frame_dict["trans"] = trans_list
        frame_dict["time"] = "{:.3f}".format(current_time)
        output.append(frame_dict)

    if config['shift_length'] > 0:
        for i in range(int(config['shift_length'])):
            output.insert(0, output[0])

    mp3_name = re.split(r'/', audio_fname)[-1]
    with open("{}{}.json".format(out_path, mp3_name[:-4]), "wb") as json_file:
        json.dump(output, json_file)


def inference(config, audio_fname):
    tf_model_fname = config['tf_model_fname']
    out_path = config['out_path']
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(config, audio)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    # num_frames = processed_audio.shape[0]
    feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                 is_training: False}

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)
        predicted_blendshapes = np.squeeze(session.run(output_decoder, feed_dict))
        # print(predicted_blendshapes.shape)
        output_sequences(predicted_blendshapes, out_path, audio_fname, config)

    tf.reset_default_graph()

    if config['trim_length'] > 0:
        wav_trim(out_path, config['trim_length'], audio_fname)


def wav_trim(output_path, trim_length, audio_fname):
    song = AudioSegment.from_wav(audio_fname)
    trim_audio = song[trim_length:]
    trim_audio.export(output_path + 'trim_' + audio_fname[-40:], format='wav')

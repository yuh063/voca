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


import os
import json
import time
import numpy as np
import tensorflow as tf

import scipy
from scipy.io import wavfile
from scipy.io.wavfile import read

from audio_handler_test import AudioHandler
from psbody.mesh import Mesh


def process_audio(ds_path, audio, sample_rate):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1

    config['sample_rate'] = 48000

    audio_arr = np.array(audio, dtype=float)
    tmp_audio = {'tmp': audio_arr}

    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio)['tmp']

def output_sequences(sequence_blendshapes, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    output = []
    current_time = time.time()
    trans_list = ["1.000", "0.000", "0.000", "0.000", "0.000", "1.000", "0.000", "0.000", "0.000", "0.000", "1.000",
                  "0.000", "0.000", "0.000", "0.000", "1.000"]
    for i in range(sequence_blendshapes.shape[0]):
        current_time += 1.0/60
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

    print(output)

    with open("{}test_3.json".format(out_path), "wb") as json_file:
        json.dump(output, json_file)


def inference(tf_model_fname, ds_fname, audio_fname, out_path):

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    # condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    # input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    num_frames = processed_audio.shape[0]
    feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                 is_training: False}

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)
        predicted_blendshapes = np.squeeze(session.run(output_decoder, feed_dict))
        output_sequences(predicted_blendshapes, out_path)

    tf.reset_default_graph()


def inference_interpolate_styles(tf_model_fname, ds_fname, audio_fname, template_fname, condition_weights, out_path):
    template = Mesh(filename=template_fname)

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    # condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    # input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    non_zeros = np.where(condition_weights > 0.0)[0]
    condition_weights[non_zeros] /= sum(condition_weights[non_zeros])

    num_frames = processed_audio.shape[0]
    output_vertices = np.zeros((num_frames, template.v.shape[0], template.v.shape[1]))

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)

        for condition_id in non_zeros:
            feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                         condition_subject_id: np.repeat(condition_id, num_frames),
                         is_training: False,
                         input_template: np.repeat(template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0)}
            predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
            output_vertices += condition_weights[condition_id] * predicted_vertices

        output_sequence_meshes(output_vertices, template, out_path)
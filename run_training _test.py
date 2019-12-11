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
import stat
import glob
import shutil
import subprocess
import ConfigParser
import tensorflow as tf

from utils.data_handler_test import DataHandler
from utils.batcher_test import Batcher
from utils.voca_model_test import VOCAModel as Model


def main():
    # TODO need to create config file in the future.
    config = {}
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29
    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    config['deepspeech_graph_fname'] = './ds_graph/output_graph.pb'
    config['sample_rate'] = 48000
    config['clear_unmatched_animation_pairs'] = True
    config['checkpoint_dir'] = './littlelights_training'
    config['num_blendshapes'] = 51
    config['expression_dim'] = 50
    config['speech_encoder_size_factor'] = 1.0
    config['absolute_reconstruction_loss'] = False
    config['velocity_weight'] = 10.0
    config['num_consecutive_frames'] = 1
    config['dataset_path'] = '/home/littlelight/voca/littlelights_dataset_v2/'
    config['audio_json_pair_file_name'] = 'audio_json_match.txt'

    config['batch_size'] = 128
    config['decay_rate'] = 1.0
    config['adam_beta1_value'] = 0.9
    config['learning_rate'] = 1e-4
    config['epoch_num'] = 50

    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    with tf.Session() as session:
        model = Model(session=session, config=config, batcher=batcher)
        model.build_graph(session)
        model.load()
        model.train()


if __name__ == '__main__':
    main()

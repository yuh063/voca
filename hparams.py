def load_hparams():
    config = {}

    ################################
    # Audio Parameters             #
    ################################
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29  # Please create new TFRecord to change this parameter
    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    config['sample_rate'] = 48000

    ################################
    # Data Parameters              #
    ################################
    config['deepspeech_graph_fname'] = './ds_graph/output_graph.pb'
    config['checkpoint_dir'] = './littlelights_training_test'
    config['dataset_path'] = '/home/littlelight/voca/littlelights_dataset_10hours/'
    config['audio_json_pair_file_name'] = 'audio_json_match.txt'
    config['clear_unmatched_animation_pairs'] = True

    ################################
    # Model Parameters             #
    ################################
    config['num_blendshapes'] = 51  # Please create new TFRecord to change this parameter
    config['expression_dim'] = 128
    config['speech_encoder_size_factor'] = 1.0
    config['absolute_reconstruction_loss'] = False
    config['velocity_weight'] = 0.5
    config['frame_rate'] = 30  # Currently only support 30 and 60
    config['num_consecutive_frames'] = 10  # Please create new TFRecord to change this parameter

    ################################
    # Optimization Hyperparameters #
    ################################
    config['batch_size'] = 16
    config['decay_rate'] = 0.9
    config['adam_beta1_value'] = 0.9
    config['learning_rate'] = 1e-3
    config['epoch_num'] = 200

    ################################
    # Output Parameter             #
    ################################
    config['tf_model_fname'] = './littlelights_training_test/checkpoints/gstep_656200.model'
    config['out_path'] = '/home/littlelight/voca/littlelights_30fps_model/'
    config['audio_path'] = '/home/littlelight/voca/littlelights_class_0_wav/'
    config['trim_length'] = 0
    config['shift_length'] = 6

    return config

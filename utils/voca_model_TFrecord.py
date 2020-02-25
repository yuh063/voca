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

from utils.base_model import BaseModel
import cv2
import os
import sys
import logging
import tempfile
from subprocess import call
import threading

import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from sklearn.manifold import TSNE

# from psbody.mesh import Mesh
# from utils.rendering import render_mesh_helper
from utils.losses import *
from utils.speech_encoder_test import SpeechEncoder
from utils.expression_layer_test import ExpressionLayer

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class VOCAModel(BaseModel):
    def __init__(self, session, batcher, config, scope='VOCA'):
        BaseModel.__init__(self, session=session, batcher=batcher, config=config, scope=scope)
        # self.template_mesh = Mesh(filename=config['template_fname'])

        self.init_placeholders = getattr(self, '_init_placeholders')
        self.build_encoder = getattr(self, '_build_encoder')
        self.build_decoder = getattr(self, '_build_decoder')

    def build_graph(self, session):
        with tf.variable_scope(self.scope):
            self.init_placeholders()

            '''
            session.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            tvars_vals = session.run(tvars)

            for var, val in zip(tvars, tvars_vals):
                print(var.name)  # Prints the name of the variable alongside its value.
            '''

            self.build_encoder()
            self.build_decoder()
            self._build_losses()
            self._init_training()
            self._build_savers(max_to_keep=10)

    def _init_placeholders(self):
        with tf.name_scope('Inputs_encoder'):
            self.speech_features = tf.placeholder(tf.float32, [None, self.config['audio_window_size'], self.config['num_audio_features'], 1], name='speech_features')
            # self.condition_subject_id = tf.placeholder(tf.int32, [None], name='condition_subject_id')
            self.is_training = tf.placeholder(tf.bool, name='is_training')
        with tf.name_scope('Target'):
            self.target_blendshapes = tf.placeholder(tf.float32, [None, self.config['num_blendshapes'], 1], name='target_blendshapes')
        # with tf.name_scope('Inputs_decoder'):
        #    self.input_template = tf.placeholder(tf.float32, [None, self.config['num_blendshapes'], 1], name='template_placeholder')

    def _build_encoder(self):
        self.output_encoder = self._build_audio_encoder()

    def _build_audio_encoder(self):
        audio_encoder = SpeechEncoder(self.config)
        # condition = tf.one_hot(indices=self.condition_subject_id, depth=self.batcher.get_num_training_subjects())
        # return audio_encoder(self.speech_features, condition, self.is_training)
        return audio_encoder(self.speech_features, self.is_training)

    def _build_decoder(self):
        expression_decoder = ExpressionLayer(self.config)
        self.output_decoder = expression_decoder(self.output_encoder)
        self.output_decoder = tf.identity(self.output_decoder, name='output_decoder')
        # self.expression_offset = expression_decoder(self.output_encoder)
        # self.output_decoder = tf.add(self.expression_offset, self.input_template, name='output_decoder')

    def _build_losses(self):
        self.rec_loss = self._reconstruction_loss()
        self.velocity_loss = self._velocity_loss()
        self.loss = self.rec_loss + self._velocity_loss()
        # self.acceleration_loss = self._acceleration_loss()
        # self.verts_reg_loss = self._verts_regularizer_loss()
        # self.loss = self.rec_loss + self.velocity_loss + self.acceleration_loss + self.verts_reg_loss

        tf.summary.scalar('loss', self.loss, collections=['train', 'validation'])
        # tf.summary.scalar('loss_validation', self.loss, collections=['validation'])
        self.t_vars = tf.trainable_variables()

    # calculate mean square error
    def _reconstruction_loss(self):
        weight = np.ones(51)
        np.put(weight, [1, 6, 10, 16, 37, 44, 48], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        with tf.name_scope('Reconstruction_loss'):
            rec_loss = reconstruction_loss(predicted=tf.reshape(self.output_decoder, [-1, 1, self.config['num_blendshapes']]),
                                           real=tf.reshape(self.target_blendshapes, [-1, 1, self.config['num_blendshapes']]),
                                           weights=weight, want_absolute_loss=self.config['absolute_reconstruction_loss'])
        tf.summary.scalar('reconstruction_loss', rec_loss, collections=['train'])
        tf.summary.scalar('reconstruction_loss', rec_loss, collections=['validation'])
        # tf.summary.scalar('reconstruction_loss_training', rec_loss, collections=['train'])
        # tf.summary.scalar('reconstruction_loss_validation', rec_loss, collections=['validation'])
        return rec_loss

    def _velocity_loss(self):
        weight = np.ones(51)
        np.put(weight, [1, 6, 10, 16, 37, 44, 48], [2, 2, 2, 2, 2, 2, 2])
        if self.config['velocity_weight'] > 0.0:
            assert(self.config['num_consecutive_frames'] >= 2)
            blendshapes_predicted = tf.reshape(self.output_decoder, [-1, self.config['num_blendshapes'], self.config['num_consecutive_frames']])
            # print(self.output_decoder.get_shape())
            velocity_loss = 0
            for i in range(self.config['num_consecutive_frames']-1):
                x1_pred = blendshapes_predicted[:, :, i+1]
                x2_pred = blendshapes_predicted[:, :, i]
                velocity_pred = x1_pred-x2_pred
                velocity_pred = tf.expand_dims(velocity_pred, 1)

                # print(velocity_pred.get_shape())

                blendshapes_target = tf.reshape(self.target_blendshapes, [-1, self.config['num_blendshapes'], self.config['num_consecutive_frames']])
                x1_target = blendshapes_target[:, :, i+1]
                x2_target = blendshapes_target[:, :, i]
                velocity_target = x1_target-x2_target
                velocity_target = tf.expand_dims(velocity_target, 1)

                # print(velocity_target.get_shape())

                with tf.name_scope('Velocity_loss{}'.format(i+1)):
                    tmp = self.config['velocity_weight']*reconstruction_loss(predicted=velocity_pred, real=velocity_target,
                                                                                       want_absolute_loss=self.config['absolute_reconstruction_loss'])
                velocity_loss += tmp
            velocity_loss = velocity_loss/(self.config['num_consecutive_frames']-1)
            tf.summary.scalar('velocity_loss_training', velocity_loss, collections=['train'])
            tf.summary.scalar('velocity_loss_validation', velocity_loss, collections=['validation'])
            return velocity_loss
        else:
            return 0.0

    '''
    def _acceleration_loss(self):
        if self.config['acceleration_weight'] > 0.0:
            assert(self.config['num_consecutive_frames'] >= 3)
            verts_predicted = tf.reshape(self.output_decoder, [-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3])
            x1_pred = tf.reshape(verts_predicted[:, -1, :], [-1, self.config['num_vertices'], 3, 1])
            x2_pred = tf.reshape(verts_predicted[:, -2, :], [-1, self.config['num_vertices'], 3, 1])
            x3_pred = tf.reshape(verts_predicted[:, -3, :], [-1, self.config['num_vertices'], 3, 1])
            acc_pred = x1_pred-2*x2_pred+x3_pred

            verts_target = tf.reshape(self.target_vertices, [-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3])
            x1_target = tf.reshape(verts_target[:, -1, :], [-1, self.config['num_vertices'], 3, 1])
            x2_target = tf.reshape(verts_target[:, -2, :], [-1, self.config['num_vertices'], 3, 1])
            x3_target = tf.reshape(verts_target[:, -3, :], [-1, self.config['num_vertices'], 3, 1])
            acc_target = x1_target-2*x2_target+x3_target

            with tf.name_scope('Acceleration_loss'):
                acceleration_loss = self.config['acceleration_weight']*reconstruction_loss(predicted=acc_pred, real=acc_target,
                                                        want_absolute_loss=self.config['absolute_reconstruction_loss'])
            tf.summary.scalar('acceleration_loss_training', acceleration_loss, collections=['train'])
            tf.summary.scalar('acceleration_loss_validation', acceleration_loss, collections=['validation'])
            return acceleration_loss
        else:
            return 0.0

    def _verts_regularizer_loss(self):
        if self.config['verts_regularizer_weight'] > 0.0:
            with tf.name_scope('Verts_regularizer_loss'):
                verts_regularizer_loss = self.config['verts_regularizer_weight']*tf.reduce_mean(tf.reduce_sum(tf.abs(self.expression_offset), axis=2))
            tf.summary.scalar('verts_regularizer_losss_training', verts_regularizer_loss, collections=['train'])
            tf.summary.scalar('verts_regularizer_loss_validation', verts_regularizer_loss, collections=['validation'])
            return verts_regularizer_loss
        else:
            return 0.0
    '''

    def _init_training(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = 5*self.batcher.get_training_size()/self.config['batch_size']
        decay_rate = self.config['decay_rate']
        if decay_rate < 1:
            self.global_learning_rate = tf.train.exponential_decay(self.config['learning_rate'], self.global_step,
                                                                   decay_steps, decay_rate, staircase=True)
        else:
            self.global_learning_rate = tf.constant(self.config['learning_rate'], dtype=tf.float32)
        tf.summary.scalar('learning_rate_training', self.global_learning_rate, collections=['train'])
        tf.summary.scalar('learning_rate_validation', self.global_learning_rate, collections=['validation'])

        self.optim = tf.train.AdamOptimizer(self.global_learning_rate, self.config['adam_beta1_value']).\
            minimize(self.loss, var_list=self.t_vars, global_step=self.global_step)

        self._init_summaries()
        tf.global_variables_initializer().run()

        self.processed_audio_training, self.blendshapes_training = self.batcher.get_training_batch()
        self.processed_audio_val, self.blendshapes_val = self.batcher.get_validation_batch()

    def _init_summaries(self):
        self.train_summary = tf.summary.merge_all('train')
        self.validation_summary = tf.summary.merge_all('validation')
        self.train_writer = tf.summary.FileWriter(os.path.join(self.config['checkpoint_dir'], 'summaries', 'train'))
        self.validation_writer = tf.summary.FileWriter(os.path.join(self.config['checkpoint_dir'], 'summaries', 'validation'))

    def train(self):
        num_train_batches = self.batcher.get_num_batches(self.config['batch_size'])
        for epoch in xrange(1, self.config['epoch_num']+1):
            for iter in xrange(num_train_batches):
                loss, g_step, summary, g_lr = self._training_step()

                if iter % 50 == 0:
                    logging.warning("Epoch: %d | Iter: %d | Global Step: %d | Loss: %.6f | Learning Rate: %.6f" % (epoch, iter, g_step, loss, g_lr))
                    self.train_writer.add_summary(summary, g_step)
                if iter % 100 == 0:
                    val_loss, val_summary = self._validation_step()
                    logging.warning("Validation loss: %.6f" % val_loss)
                    self.validation_writer.add_summary(val_summary, g_step)

            if epoch % 10 == 0:
                self._save(g_step)

        '''        
            if epoch % 25 == 0:
                self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 'training_epoch_%d_iter_%d' % (epoch, iter))
                                       , data_specifier='training')
                self._render_sequences(out_folder=os.path.join(self.config['checkpoint_dir'], 'videos', 'validation_epoch_%d_iter_%d' % (epoch, iter))
                                       , data_specifier='validation')
        '''
        self._finish()

    def _training_step(self):
        processed_audio, blendshapes = self.session.run([self.processed_audio_training, self.blendshapes_training])

        feed_dict = {self.speech_features: np.expand_dims(processed_audio, -1),
                     # self.condition_subject_id: np.array(subject_idx),
                     self.is_training: True,
                     # self.input_template: np.expand_dims(templates, -1),
                     self.target_blendshapes: np.expand_dims(blendshapes, -1)}

        loss, g_step, summary, g_lr, _ = self.session.run([self.loss, self.global_step, self.train_summary, self.global_learning_rate, self.optim], feed_dict)
        return loss, g_step, summary, g_lr

    def _validation_step(self):
        processed_audio, blendshapes = self.session.run([self.processed_audio_val, self.blendshapes_val])

        # Compute validation error conditioned on all training subjects and return mean over all
        # num_training_subjects = self.batcher.get_num_training_subjects()
        # conditions = np.reshape(np.repeat(np.arange(num_training_subjects)[:,np.newaxis],
        #                                  repeats=self.config['num_consecutive_frames']*self.config['batch_size'], axis=-1), [-1,])

        feed_dict = {self.speech_features: np.expand_dims(processed_audio, -1),
                    # self.condition_subject_id: conditions,
                    self.is_training: False,
                    # self.input_template: np.expand_dims(np.repeat(templates, repeats=num_training_subjects, axis=0), -1),
                    self.target_blendshapes: np.expand_dims(blendshapes, -1)}
        loss, summary = self.session.run([self.loss, self.validation_summary], feed_dict)
        return loss, summary

    '''
    def _render_sequences(self, out_folder, run_in_parallel=True, data_specifier='validation'):
        print('Render %s sequences' % data_specifier)
        if run_in_parallel:
            self.threads.append(threading.Thread(target=self._render_helper, args=(out_folder, data_specifier)))
            self.threads[-1].start()
        else:
            self._render_helper(out_folder, data_specifier)

    def _render_helper(self, out_folder, data_specifier):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        if data_specifier == 'training':
            raw_audio, processed_audio, vertices, templates, subject_idx = self.batcher.get_training_sequences_in_order(
                self.num_render_sequences)
            #Render each training sequence with the corresponding condition
            condition_subj_idx = [[idx] for idx in subject_idx]
        elif data_specifier == 'validation':
            raw_audio, processed_audio, vertices, templates, subject_idx = self.batcher.get_validation_sequences_in_order(
                self.num_render_sequences)
            #Render each validation sequence with all training conditions
            num_training_subjects = self.batcher.get_num_training_subjects()
            condition_subj_idx = [range(num_training_subjects) for idx in subject_idx]
        else:
            raise NotImplementedError('Unknown data specifier %s' % data_specifier)

        for i_seq in range(len(raw_audio)):
            conditions = condition_subj_idx[i_seq]
            for condition_idx in conditions:
                condition_subj = self.batcher.convert_training_idx2subj(condition_idx)
                video_fname = os.path.join(out_folder, '%s_%03d_condition_%s.mp4' % (data_specifier, i_seq, condition_subj))
                self._render_sequences_helper(video_fname, raw_audio[i_seq], processed_audio[i_seq], templates[i_seq], vertices[i_seq], condition_idx)

    def _render_sequences_helper(self, video_fname, seq_raw_audio, seq_processed_audio, seq_template, seq_verts, condition_idx):
        def add_image_text(img, text):
            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (img.shape[1] - textsize[0]) / 2
            textY = textsize[1] + 10
            cv2.putText(img, '%s' % (text), (textX, textY), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        num_frames = seq_verts.shape[0]
        tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.wav', dir=os.path.dirname(video_fname))
        wavfile.write(tmp_audio_file.name, seq_raw_audio['sample_rate'], seq_raw_audio['audio'])

        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=os.path.dirname(video_fname))
        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            # writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'DIVX'), 60, (4000, 800), True)
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (1600, 800), True)
        else:
            print('cv2 >= 3')
            # writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'DIVX'), 60, (4000, 800), True)
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (1600, 800), True)

        feed_dict = {self.speech_features: np.expand_dims(np.stack(seq_processed_audio), -1),
                     self.condition_subject_id: np.repeat(condition_idx, num_frames),
                     self.is_training: False,
                     self.input_template: np.repeat(seq_template[np.newaxis,:,:,np.newaxis], num_frames, axis=0)}

        predicted_vertices, predicted_offset = self.session.run([self.output_decoder, self.expression_offset], feed_dict)
        predicted_vertices = np.squeeze(predicted_vertices)
        center = np.mean(seq_verts[0], axis=0)

        for i_frame in range(num_frames):
            gt_img = render_mesh_helper(Mesh(seq_verts[i_frame], self.template_mesh.f), center)
            add_image_text(gt_img, 'Captured data')
            pred_img = render_mesh_helper(Mesh(predicted_vertices[i_frame], self.template_mesh.f), center)
            add_image_text(pred_img, 'VOCA prediction')
            img = np.hstack((gt_img, pred_img))
            writer.write(img)
        writer.release()

        cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
            tmp_audio_file.name, tmp_video_file.name, video_fname)).split()
        call(cmd)
        '''

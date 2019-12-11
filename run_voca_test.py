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
import glob
import argparse
from utils.inference_test import inference


def main():
    tf_model_fname = './littlelights_training/checkpoints/gstep_129030.model'
    ds_fname = './ds_graph/output_graph.pb'
    audio_fname = './littlelights_output/test_3.wav'
    # template_fname = args.template_fname
    # condition_idx = args.condition_idx
    out_path = './littlelights_output/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    inference(tf_model_fname, ds_fname, audio_fname, out_path)


if __name__ == '__main__':
    main()
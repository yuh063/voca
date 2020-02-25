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

from hparams import load_hparams
from glob import glob
from utils.inference_test import inference


def main():
    config = load_hparams()
    audio_files = config['audio_path'] + '*.wav'
    audio_fnames = glob(audio_files)
    for audio_fname in audio_fnames:
        inference(config, audio_fname)


if __name__ == '__main__':
    main()

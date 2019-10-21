'''
This script is used to read multiple .mat files and convert to corresponding
.png files (or .jpg or anything else)
'''

import os
from scipy.io import loadmat
from scipy.misc import imsave

os.getcwd()

src_dir = './Datasets/2-UMichigan-corridor/mat/'
dst_dir = './Datasets/2-UMichigan-corridor/png/'

for i, filename in enumerate(os.listdir(src_dir)):
    # get image name from .mat file
    name_split = filename.split(".")
    name = name_split[0]
    print(name)

    # load .mat file and convert to .png file
    mat = loadmat(src_dir + filename)
    img = mat['ground_truth']
    img[img != -1] = 0  # force not-ground-pixels BLACK
    img[img == -1] = 1  # force ground-pixels WHITE

    # save .png file
    imsave(dst_dir + name + '.png', img)

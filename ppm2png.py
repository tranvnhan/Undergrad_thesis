'''
This script is used to read multiple .ppm files and convert to corresponding
.png files (or .jpg or anything else)
'''

import os
from PIL import Image

os.getcwd()

src_dir = './Datasets/2-UMichigan-corridor/ppm/'
dst_dir = './Datasets/2-UMichigan-corridor/png/'

for i, filename in enumerate(os.listdir(src_dir)):
    # get image name from .mat file
    name_split = filename.split(".")
    name = name_split[0]

    # load .ppm file and convert to .png file
    img = Image.open(src_dir + filename)
    img.save(dst_dir + name + '.png')

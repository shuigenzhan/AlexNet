import warnings
from shutil import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def mkdir(file):
    if not os.path.exists(file):
        os.makedirs(file)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    path = './dataset/flower_photos/'
    flower_class = [cla for cla in os.listdir(path)]
    for cla in flower_class:
        mkdir('./dataset/train/' + cla)
    for cla in flower_class:
        mkdir('./dataset/val/' + cla)

    VAL_RATION = 0.1
    for cla in flower_class:
        file = path + cla + '/'
        images = os.listdir(file)
        val_num = int(len(images) * VAL_RATION)
        val_images = images[:val_num]
        train_images = images[val_num:]
        for image in val_images:
            val_path = file + image
            new_path = os.path.join('./dataset/val/', cla, image)
            copy(val_path, new_path)
        for image in train_images:
            train_path = file + image
            new_path = os.path.join('./dataset/train/', cla, image)
            copy(train_path, new_path)
        print(f'successfully copy {cla} class')




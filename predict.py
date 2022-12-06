import argparse
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image
from model.model import AlexNet
from torchvision import transforms

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model', type=str, help="Model weights.")
    parser.add_argument('-test_fn', type=str, default='./dataset/test/rose.jpg', help="Test data.")
    parser.add_argument('-nc', '--num_class', type=int, default=5, help="Model weights.")

    args = parser.parse_args()

    data_transform = transforms.Compose([transforms.Resize((227, 227)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    pic_path = args.test_fn
    image = Image.open(pic_path)
    image = data_transform(image)
    class_dict = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

    model = AlexNet(num_class=args.num_class)
    model.load_state_dict(torch.load(args.model))

    model.eval()
    with torch.no_grad():
        y_hat = model(image)
        y_hat = torch.softmax(y_hat)
        print(f'The pic is {class_dict[torch.argmax(y_hat)]}')

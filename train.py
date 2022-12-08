import argparse
import warnings
import os

import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model.model import AlexNet

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-drop', '--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('-l2', '--l2_reg', type=float, default=1e-4, help="L2 regularization coefficient.")
    parser.add_argument('-lr', type=float, default=0.0002, help="Initial learning rate.")
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Number of epochs to train.")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('-train_fn', type=str, default='./dataset/train/', help='train dataset')
    parser.add_argument('-valid_fn', type=str, default='./dataset/val/', help='valid dataset')
    parser.add_argument('-nc', '--num_class', type=int, default=5,  help="number class.")
    args = parser.parse_args()

    writer = SummaryWriter('./log/')

    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(227),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((227, 227)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    batch_size = args.batch_size
    train_set = datasets.ImageFolder(root=args.train_fn, transform=data_transform['train'])
    val_set = datasets.ImageFolder(root=args.valid_fn, transform=data_transform['val'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    network = AlexNet(num_class=args.num_class, dropout=args.dropout)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=args.lr)
    epochs = args.epochs
    train_step = len(train_loader)
    val_num = len(val_loader)
    best_acc = 0.0
    for epoch in range(epochs):
        network.train()
        step = 1
        for images, labels in train_loader:
            optimizer.zero_grad()
            y_hat = network(images)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss, step)
            writer.add_images('images', images, step)
            print(f'Epoch: [{epoch + 1} / {epochs}], step: [{step} / {train_step}], loss: [{loss:.4f}]')
            step += 1

        network.eval()
        accuracy = 0.0
        for images, labels in val_loader:
            y_hat = network(images)
            prediction = torch.max(y_hat, dim=1)[1]
            accuracy += (prediction == labels).sum().item()
        accuracy /= val_num
        print(f'Epoch: [{epoch + 1} / {epochs}], accuracy: [{accuracy:.4f}]')
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(network.state_dict(), './output/{}_{}'.format(epoch, accuracy))
    writer.close()

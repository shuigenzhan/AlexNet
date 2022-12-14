import argparse
import warnings

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from model.model import AlexNet

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate.")
    parser.add_argument('--lr', type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size.")
    parser.add_argument('--train_fn', type=str, default='./CIFAR10/', help='train dataset')
    parser.add_argument('--test_fn', type=str, default='./CIFAR10/', help='test dataset')
    parser.add_argument('--num_class', type=int, default=10, help="number class.")
    parser.add_argument('--valid_ration', type=float, default=0.1, help="ration of valid set")
    args = parser.parse_args()

    # writer = SummaryWriter('./log/')

    data_transform = transforms.Compose(
        [transforms.Resize((227, 227), interpolation=InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))])

    batch_size = args.batch_size
    VAL_RATION = args.valid_ration
    dataset = datasets.CIFAR10(root=args.train_fn, train=True, transform=data_transform, download=True)
    train_size = int((1 - VAL_RATION) * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    network = AlexNet(num_class=args.num_class, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=network.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    train_step = len(train_loader)
    valid_num = len(valid_set)
    best_acc = 0.0

    for epoch in range(args.epochs):
        network.train()
        step = 1
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            y_hat = network(images)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('loss', loss, step)
            # writer.add_images('images', images, step)
            # writer.add_graph(network, images)
            if step % 10 == 0:
                print(f'Epoch: [{epoch + 1} / {args.epochs}], step: [{step} / {train_step}], loss: [{loss:.4f}]')
            step += 1

        network.eval()
        with torch.no_grad():
            accuracy = 0.0
            loss = 0.0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                y_hat = network(images)
                loss += loss_fn(y_hat, labels)
                prediction = torch.max(y_hat, dim=1)[1]
                accuracy += (prediction == labels).sum().item()
            accuracy /= valid_num
            print(f'Epoch: [{epoch + 1} / {args.epochs}], loss: [{loss / valid_num:.4f}], accuracy: [{accuracy:.4f}]')
            if accuracy > best_acc:
                best_acc = accuracy
                print('Saving model to ./output/')
                torch.save(network.state_dict(), './output/weights_{}_{}.ckpt'.format(loss, accuracy))
    # writer.close()

import argparse
import warnings

import torch

from torch.utils.data import DataLoader
from model.model import AlexNet
from torchvision import transforms, datasets

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str, help="Model weights.")
    parser.add_argument('--test_fn', type=str, default='./CIFAR10/', help="Test data.")
    parser.add_argument('--num_class', type=int, default=10, help="Model weights.")

    args = parser.parse_args()

    data_transform = transforms.Compose(
        [transforms.Resize((227, 227), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(),
         transforms.Normalize(mean=(0.4738, 0.4730, 0.4299), std=(0.2547, 0.2524, 0.2690))])
    dataset = datasets.CIFAR10(root=args.test_fn, train=False, transform=data_transform, download=True)
    test_loader = DataLoader(dataset, batch_size=64)
    test_num = len(dataset)

    model = AlexNet(num_class=args.num_class)
    model.load_state_dict(torch.load(args.model_name))

    model.eval()
    with torch.no_grad():
        accuracy = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            y_hat = model(images)
            prediction = torch.max(y_hat, dim=1)[1]
            accuracy += (prediction == labels).sum().item()
        accuracy /= test_num
        print(f'The test accuracy is [{accuracy:.4f}]')

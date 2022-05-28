import os
from urllib import parse
import torch
from torch.distributed.distributed_c10d import get_rank, get_world_size 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

import argparse
import torch.distributed as dist 
import torch.multiprocessing as mp
import dist_utils
import time
# import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)

        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


def train(model, train_loader, criterion, optimizer, num_epochs=2):
    # train
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.
    model.train()
    # Loss_average = []
    dist_utils.init_parameters(model)

    starttime = time.time()  # 当前时间
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            # cuda
            inputs, labels = inputs.to(dist_utils.get_local_rank()), labels.to(dist_utils.get_local_rank())

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            # Loss_each_epoch.append(loss.item())

            # pipeline
            optimizer.zero_grad()
            loss.backward()
            # averge the gradients of model parameters
            dist_utils.allreduce_average_gradients(model)

            optimizer.step()
            loss_total += loss.item()

            if i % 20 == 19:    # print every 2000 mini-batches
                print('Device: %d epoch: %d, iters: %5d, loss: %.3f' % (dist_utils.get_local_rank(), epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0

    print("Training Finished!")
    endtime = time.time()#结束时间
    train_time = endtime-starttime
    print("Training time: {}".format(train_time))


def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--gpu', default="0", type=str, help='GPU ID')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # get args
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # args.cuda = True  # cuda

    # distributed initilization
    dist_utils.dist_init(args.n_devices, args.rank)
    # construct the model
    model = Net(in_channels=1, num_classes=10)
    model.to(dist_utils.get_local_rank())  # cuda

    # construct the dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.MNIST("./data/", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST("./data/", train=False, download=True, transform=transform)

    from torch.utils.data.distributed import DistributedSampler

    sampler = DistributedSampler(dataset=train_set, num_replicas=args.n_devices, rank=args.rank)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # construct the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train(model, train_loader, criterion, optimizer)
    test(model, test_loader)

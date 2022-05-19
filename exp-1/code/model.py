import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torch.optim as optim
from MyOptimizer import MyOptimizer, MyOptimizerAdam

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

def train(model, train_loader, optimizer, criterion, num_epochs=1):
    # train

    ###########################################################

    # Input:
    # model: 指定训练的模型
    # train_loader: 训练数据的DataLoader对象
    # optimizer: 需要在本次实验中实现的优化器对象
    # criterion: 训练中使用的损失函数
    # num_epochs: 训练的总epoch数

    ###########################################################
    
    print("Start training ...")
    loss_total = 0.
    model.train()
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(train_loader):
            # with dist_autograd.context() as context_id:
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # pipeline
            # dist_autograd.backward(context, [loss])
            # optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            loss_total += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0
    
    print("Training Finished!")

def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def main():
    # construct the model
    model = Net(in_channels=1, num_classes=10)
    model.cuda()

    DATA_PATH = ""

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    #optimizer = MyOptimizer(model.parameters(), lr=0.01)
    #optimizer = MyOptimizerAdam(model.parameters(), lr=0.01, b1=0.9, b2=0.999)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(model, train_loader, optimizer, criterion)
    test(model, test_loader)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
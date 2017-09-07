#!/usr/bin/python

import os

import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

#
# DATA HANDLING
#

batch_size = 4

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/data/cifar10'), train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img, norm=True):
    if norm:
        img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

# print(iter(trainloader).next())

#
# NEURAL NETWORK
#

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#
# TRAINING
#

# Loading/saving taken from https://github.com/pytorch/pytorch/blob/761d6799beb3afa03657a71776412a2171ee7533/docs/source/notes/serialization.rst
def save_model(model, path):
    return torch.save(model.state_dict(), path)

def load_model(the_model, path):
    model_placeholder = torch.load(path)
    the_model.load_state_dict(torch.load(PATH))

net = Net()
if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epochs = 2
for epoch in range(n_epochs):
    running_loss = 0.0
    # TODO When wouldn't you start at 0? Upon inspection, they appear to be the
    # same.
    for i,d in enumerate(trainloader, 0):
        inputs, labels = d
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        logfreq = 2000
        if i % logfreq == logfreq-1:
            print('epoch %d iter %5d loss %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# TODO factor out these paths
save_model(net, os.path.expanduser("~/results/pytorch-tutorial/foobar"))

# TODO load model

#
# TEST
#

# overall accuracy
correct = 0
total = 0
for d in testloader:
    images, labels = d
    outputs = net(Variable(images))
    _,predicted = torch.max(outputs.data, 1)

    correct += (predicted == labels).sum()
    total += labels.size(0)

print('Accuracy on test set (10,000 images): %d %%' % (100 * correct / total))

# class-wise accuracy
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for d in testloader:
    images, labels = d
    outputs = net(Variable(images))
    _,predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s: %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

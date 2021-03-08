import pandas as pd
import numpy as np
import torchvision
import torch
from torchvision import transforms
import torch.nn as nn


# Convolutional neural network
class convnet(nn.Module):
    def __init__(self):
        super(convnet,self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(8*8*32, n_class)
        #self.fc = nn.Linear(20, n_class)
    def forward(self, x):

        out1 = self.layer1(x)
       # print(out1.shape)
        out2 = self.layer2(out1)
        out2 = out2.reshape(out2.size(0), -1)
      #  print(out2.shape)
        y    = self.fc(out2)
        return y


if __name__ == "__main__":
    # Params
    batch_size = 64
    n_class = 10
<<<<<<< HEAD
    lr = 0.0001
    num_epochs = 15
=======
    lr = 0.001
    num_epochs = 10
>>>>>>> 6f36b05123a78b69dbb7dbc983f146b4fda5b42c

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model CNN
    convmodel = convnet()
    convmodel.to(device)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # Optim
<<<<<<< HEAD
    optimizer_fn = torch.optim.SGD(convmodel.parameters(), lr=0.001, momentum=0.9)
=======
    optimizer_fn = torch.optim.Adam(convmodel.parameters(), lr=0.0001)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer_fn, 5, gamma=0.5)
>>>>>>> 6f36b05123a78b69dbb7dbc983f146b4fda5b42c

  #  optimizer_fn = torch.optim.Adam(convmodel.parameters(), lr=0.0001)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer_fn, 5, gamma=1)
      
    num_steps = len(trainloader)

    for i in range(num_epochs):
        convmodel.train()
        for j, (imgs, lbls) in enumerate(trainloader):
            optimizer_fn.zero_grad()
            #print(imgs.shape)
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            out = convmodel(imgs)
            loss_val = loss_fn(out, lbls)
            loss_val.backward()
            optimizer_fn.step()
            if (j + 1) % 2 == 0:
                print('Train, Epoch [{}/{}] Step [{}/{}] Loss: {:.4f}'.
                      format(i + 1, num_epochs, j + 1, num_steps, loss_val.item()))
        lr_sch.step()
            # if j == 10:
            #     break

    convmodel.eval()
    corrects = 0
    num_steps = len(testloader)
    for j, (imgs, lbls) in enumerate(testloader):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        out = convmodel(imgs)
        predicted = torch.argmax(out, 1)
        corrects += torch.sum(predicted == lbls)
        print('Step [{}/{}] Acc {:.4f}: '.format(j+1, num_steps, 100.*corrects/((j+1)*batch_size)))
<<<<<<< HEAD
=======

    # torch.save()
>>>>>>> 6f36b05123a78b69dbb7dbc983f146b4fda5b42c

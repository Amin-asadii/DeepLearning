import torch
import torchvision
from torchvision import transforms
import torch.nn as nn



# Params
batch_size = 64
n_class = 10
lr = 0.001
num_epochs = 100


# Load Custom Dataset
train_dataset = torchvision.datasets.ImageFolder('E:/Work/Uni/PHD/Deep Learning/CIFAR/dbMNIST/train',
                                                 transform=transforms.ToTensor())
valid_dataset = torchvision.datasets.ImageFolder('E:/Work/Uni/PHD/Deep Learning/CIFAR/dbMNIST/validation',
                                                 transform=transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder('E:/Work/Uni/PHD/Deep Learning/CIFAR/dbMNIST/test',
                                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)




# Convolutional neural network
class convnet(nn.Module):
    def __init__(self):
        super(convnet,self).__init__()
        # self.conv1 = nn.Conv2d(1, 16, 3, 1, 2)
        # self.BatchN1 = nn.BatchNorm2d(16)
        # self.relu1 = nn.ReLU()
        # self.maxp1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(16, 32, 3, 1, 2)
        # self.BatchN2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU()
        # self.maxp2 = nn.MaxPool2d(2, 2)
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 2),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        self.fc = nn.Linear(8*8*32, n_class)
    def forward(self, x):
        # a = self.conv1(x)
        # a2 = self.BatchN1(a)
        # a3 = self.relu1(a2)
        # a4 = self.maxp1(a3)
        # a = self.conv1(x)
        # a2 = self.BatchN1(a)
        # a3 = self.relu1(a2)
        # y = self.maxp1(a3)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2 = out2.reshape(out2.size(0), -1)
        y    = self.fc(out2)
        return y


# Model CNN
convmodel = convnet()

# loss
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer_fn = torch.optim.Adam(convmodel.parameters(), lr=lr)

# LR
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer_fn, 1, gamma=0.5)

num_steps = len(train_loader)
valid_num_steps = len(valid_loader)
for i in range(num_epochs):
    convmodel.train()
    # if condition on d_loss < 0.2:
    #     lr_sch.step()
    print(lr_sch.get_lr())
    for j, (imgs, lbls) in enumerate(train_loader):
        out = convmodel(imgs)
        loss_val = loss_fn(out, lbls)
        optimizer_fn.zero_grad()
        loss_val.backward()
        optimizer_fn.step()
        if (j+1) % 2 == 0:
            print('Train, Epoch [{}/{}] Step [{}/{}] Loss: {:.4f}'.
                  format(i+1, num_epochs, j+1, num_steps, loss_val.item()))
        if j == 10:
            break
    convmodel.eval()
    corrects = 0
    for k, (imgs, lbls) in enumerate(valid_loader):
        out = convmodel(imgs)
        loss_val = loss_fn(out, lbls)
        predicted = torch.argmax(out, 1)
        corrects += torch.sum(predicted == lbls)
        print('Validation, Step [{}/{}] Loss: {:.4f} Acc: {:.4f} '.format(k + 1, valid_num_steps, loss_val.item(), 100. * corrects / ((k + 1) * batch_size)))

convmodel.eval()
corrects = 0
num_steps = len(test_loader)
for j, (imgs, lbls) in enumerate(test_loader):
    out = convmodel(imgs)
    predicted = torch.argmax(out, 1)
    corrects += torch.sum(predicted == lbls)
    print('Step [{}/{}] Acc {:.4f}: '.format(j+1, num_steps, 100.*corrects/((j+1)*batch_size)))

torch.save()
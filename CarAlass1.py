
import torch
import torchvision
from torchvision import transforms
import  torch.nn  as nn
from PIL import Image
import cv2
#print("GeeksForGeeks")
#print("Your OpenCV version is: " + cv2.__version__)
#image_cv2 = cv2.imread('E:/AI_2/Deep Learning/DEEP LEARNING Hosam/Car Detect Project/IranianVehiclesPicture\\Pride\\111\\AYbcbeG4.1.jpg')
#print(type(image_cv2))
#print(image_cv2.shape, "\n")
#print(image_cv2)
#resized_image = cv2.resize(image_cv2, (600, 188))

# Show Imaage in a window
#cv2.imshow("Memes", resized_image)
#cv2.imwrite("generated_memes.jpg", resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


batch_size = 64
n_class = 5
lr = 0.001
num_epochs = 15

#transform = transforms.Compose([ transforms.Resize((32, 32)), transforms.ToTensor(),
                           #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.ImageFolder('E:/AI_2/Deep Learning/DEEP LEARNING Hosam/Car Detect Project/Car Divar/Divar Image/train',
                                                 transform=transforms.ToTensor())


testset = torchvision.datasets.ImageFolder('E:/AI_2/Deep Learning/DEEP LEARNING Hosam/Car Detect Project/Car Divar/Divar Image/test',
                                                 transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                          batch_size=batch_size,
                                          shuffle=True)

testloader  = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=batch_size,
                                          shuffle=True)

classes = ('nissan', 'pegu', 'pride', 'samand', 'toyota')


#COD
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
        self.fc = nn.Linear(80000, n_class)
    def forward(self, x):

        out1 = self.layer1(x)
       # print(out1.shape)
        out2 = self.layer2(out1)
        out2 = out2.reshape(out2.size(0), -1)
        print(out2.shape)
        y    = self.fc(out2)
        return y


if __name__ == "__main__":
    # Params


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])





    # Model CNN
    convmodel = convnet()
    convmodel.to(device)

    # loss
    loss_fn = nn.CrossEntropyLoss()

    # Optim
    optimizer_fn = torch.optim.Adam(convmodel.parameters(), lr=lr)
    lr_sch = torch.optim.lr_scheduler.StepLR(optimizer_fn, 5, gamma=0.5)

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

    # torch.save()
!pip install torchsummary

from torchsummary import summary

import torch 

from torch import nn,optim

import torch.nn.functional as F

import torchvision

from torchvision import datasets,transforms

from torchvision.utils import save_image 

import warnings 

warnings.filterwarnings("ignore")
batch_size=32

transform=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),

                              transforms.ToTensor(),

                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)



classes = ('Plane', 'Car', 'Bird', 'Cat','Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
class CNN_classifier(nn.Module):

    def __init__(self):

        super(CNN_classifier,self).__init__()

        #Input is 32x32x3

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=4,stride=2,padding=1,padding_mode='zeros')

        self.gn1   = nn.GroupNorm(8,16)

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode='zeros')

        self.gn2   = nn.GroupNorm(32,64)

        self.conv3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        self.gn3  = nn.GroupNorm(32,64)

        self.conv4 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1,padding=0,padding_mode='zeros')

        self.gn4   = nn.GroupNorm(16,32)

        self.flatten = nn.Flatten()

        self.fdn = nn.Linear(512,10)

        self.drop = nn.Dropout2d(p=0.5)

        self.pool = nn.MaxPool2d(kernel_size=4,stride=2,padding=1)

    def forward(self,x):

        alpha=0.02

        x=F.leaky_relu(self.gn1(self.conv1(x)),alpha)

       # x=self.drop(x)

        x=self.pool(x)

        x=F.leaky_relu(self.gn2(self.conv2(x)),alpha)

        #x=self.drop(x)

        x=F.leaky_relu(self.gn3(self.conv3(x)),alpha)

        #x=self.drop(x)

        x=self.fdn(self.flatten(F.leaky_relu(self.gn4(self.conv4(x)),alpha)))

        x=F.softmax(x)

        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Classifier = CNN_classifier().to(device)

print(Classifier)

print(summary(Classifier,input_size=(3,32,32)))

epochs=50

beta1=0.99

beta2=0.999

lr=1e-4

optimizer = optim.Adam(Classifier.parameters(),lr=lr,betas=(beta1,beta2))

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=int(epochs/10))

loss_fn= nn.CrossEntropyLoss()
training_loss=0

for epoch in range(0,epochs):

    steps=0

    for step,(image,labels) in enumerate(trainloader):

        image=image.to(device)

        labels=labels.to(device)

        optimizer.zero_grad()

        pred_labels = Classifier(image)

        error = loss_fn(pred_labels,labels)

        error.backward()

        optimizer.step()

        training_loss+=error.item()

        if(step%len(trainloader)==0):

            print('[%d/%d] [%d/%d] Classification Loss: %4f'%(epoch+1,epochs,step,len(trainloader),error.item()))



    scheduler.step(error)
Classifier.eval()

total_error=0

correct=0

total=0

classes_correct = list(0.0 for i in range(0,len(classes)))

classes_true = list(0.0 for i in range(0,len(classes)))

for ii,(image,labels) in enumerate(testloader):

    image=image.to(device)

    labels=labels.to(device)

    pred = Classifier(image)

    error = loss_fn(pred,labels)

    total_error+=error.item()

    _,predicted = torch.max(pred.data,1)

    temp = (predicted==labels).squeeze()

    for i in range(0,len(classes)):

        label=labels[i]

        classes_correct[label]+=temp[i].item()

        classes_true[label]+=1

        

for i in range(0,len(classes)):

    print("Accuracy of",classes[i],"is",100*classes_correct[i]/classes_true[i])
!pip install torchsummary

!pip install tqdm

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd

import cv2

from PIL import Image

import os

import numpy as np

from matplotlib import pyplot as plt

from tqdm import notebook



import torch

import torchvision.models as models

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader, Dataset

from torchsummary import summary



import math



from sklearn.model_selection import train_test_split
labels = pd.read_csv("../input/histopathologic-cancer-detection/train_labels.csv")

sub = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')

train_path = '../input/histopathologic-cancer-detection/train/'

test_path = '../input/histopathologic-cancer-detection/test/'



%matplotlib inline 



print(labels.shape)

#train['label'].mean() # means 40% of data is positive

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device
#Splitting data into train and val

train, test = train_test_split(labels, stratify=labels.label, test_size=0.1)

len(train), len(test)
class MyDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name+'.tif')

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
import torchvision

import torchvision.transforms as transforms



batch_size = 512



trans = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(100),

    transforms.RandomCrop(96),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans)

dataset_test = MyDataset(df_data=test, data_dir=train_path, transform=trans)



trainloader = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

testloader = DataLoader(dataset = dataset_test, batch_size=batch_size//2, shuffle=True, num_workers=8, pin_memory=True )
img, label = dataset_train[0]

print('Shape of img:', img.shape)

print('Range of img:', img.min().item(), 'to', img.max().item())

print('\nType of img:', type(img))

print('Type of items in img:', img.dtype)
#Referenced works:

#https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py

class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):

        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(

            nn.Linear(channel, channel // reduction, bias=False),

            nn.ReLU(inplace=True),

            nn.Linear(channel // reduction, channel, bias=False),

            nn.Sigmoid()

        )



    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
class custom_arch(nn.Module):

    def __init__(self):

        super().__init__()        

        

        self.conv1=nn.Sequential(

            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=1,stride=1,padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2))



        self.conv2=nn.Sequential(

            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2))

        

        self.conv3=nn.Sequential(

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2))

        

        self.conv4=nn.Sequential(

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2))

        

        self.conv5=nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(512),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2))

        

        self.dropout2d = nn.Dropout2d()

        

        

        self.fc=nn.Sequential(

            nn.Linear(512*3*3,1024),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(1024,512),

            nn.Dropout(0.4),

            nn.Linear(512, 2),

            nn.Sigmoid())



                

    def forward(self, x):

        x=self.conv1(x)

        x=self.conv2(x)

        x=self.conv3(x)

        x=self.conv4(x)

        x=self.conv5(x)

        #print(x.shape) #<-- Life saving debugging step :D

        x=x.view(x.shape[0],-1)

        x=self.fc(x)

        return x



#summary(Network1(),(3,96,96), device="cpu")
#Referenced works

#https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py



from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models import ResNet







def conv3x3(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)





class SEBasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None,

                 *, reduction=16):

        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)

        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SELayer(planes, reduction)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        residual = x

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.se(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





class SEBottleneck(nn.Module):

    expansion = 4



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None,

                 *, reduction=16):

        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,

                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(planes * 4, reduction)

        self.downsample = downsample

        self.stride = stride



    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)

        out = self.se(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out += residual

        out = self.relu(out)



        return out





def se_resnet18(num_classes=1_000):

    """Constructs a ResNet-18 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet34(num_classes=1_000):

    """Constructs a ResNet-34 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet50(num_classes=1_000, pretrained=False):

    """Constructs a ResNet-50 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    if pretrained:

        model.load_state_dict(load_state_dict_from_url(

            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"))

    return model





def se_resnet101(num_classes=1_000):

    """Constructs a ResNet-101 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





def se_resnet152(num_classes=1_000):

    """Constructs a ResNet-152 model.



    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model





class CifarSEBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, reduction=16):

        super(CifarSEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SELayer(planes, reduction)

        if inplanes != planes:

            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),

                                            nn.BatchNorm2d(planes))

        else:

            self.downsample = lambda x: x

        self.stride = stride



    def forward(self, x):

        residual = self.downsample(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.se(out)



        out += residual

        out = self.relu(out)



        return out





class CifarSEResNet(nn.Module):

    def __init__(self, block, n_size, num_classes=10, reduction=16):

        super(CifarSEResNet, self).__init__()

        self.inplane = 16

        self.conv1 = nn.Conv2d(

            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplane)

        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(

            block, 16, blocks=n_size, stride=1, reduction=reduction)

        self.layer2 = self._make_layer(

            block, 32, blocks=n_size, stride=2, reduction=reduction)

        self.layer3 = self._make_layer(

            block, 64, blocks=n_size, stride=2, reduction=reduction)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(64, num_classes)

        self.initialize()



    def initialize(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride, reduction):

        strides = [stride] + [1] * (blocks - 1)

        layers = []

        for stride in strides:

            layers.append(block(self.inplane, planes, stride, reduction))

            self.inplane = planes



        return nn.Sequential(*layers)



    def forward(self, x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)



        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)



        return x





class CifarSEPreActResNet(CifarSEResNet):

    def __init__(self, block, n_size, num_classes=10, reduction=16):

        super(CifarSEPreActResNet, self).__init__(

            block, n_size, num_classes, reduction)

        self.bn1 = nn.BatchNorm2d(self.inplane)

        self.initialize()



    def forward(self, x):

        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)



        x = self.bn1(x)

        x = self.relu(x)



        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)





def se_resnet20(**kwargs):

    """Constructs a ResNet-18 model.



    """

    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)

    return model





def se_resnet32(**kwargs):

    """Constructs a ResNet-34 model.



    """

    model = CifarSEResNet(CifarSEBasicBlock, 5, **kwargs)

    return model





def se_resnet56(**kwargs):

    """Constructs a ResNet-34 model.



    """

    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)

    return model





def se_preactresnet20(**kwargs):

    """Constructs a ResNet-18 model.



    """

    model = CifarSEPreActResNet(CifarSEBasicBlock, 3, **kwargs)

    return model





def se_preactresnet32(**kwargs):

    """Constructs a ResNet-34 model.



    """

    model = CifarSEPreActResNet(CifarSEBasicBlock, 5, **kwargs)

    return model





def se_preactresnet56(**kwargs):

    """Constructs a ResNet-34 model.



    """

    model = CifarSEPreActResNet(CifarSEBasicBlock, 9, **kwargs)

    return model
def build_network_custom(freeze_se = False):

    custom = custom_arch()

    if freeze_se == True:

        #Add SELayers

        custom.conv1.add_module("SELayer", SELayer(32, 16))

        custom.conv2.add_module("SELayer",SELayer(64, 16))

        custom.conv3.add_module("SELayer",SELayer(128, 16))

        custom.conv4.add_module("SELayer",SELayer(256, 16))

        custom.conv5.add_module("SELayer",SELayer(512, 16))

        #Freeze half of layers

        for name, param in custom.named_parameters():  

            if any(name.startswith(ext) for ext in ['conv1', 'conv2', 'conv3.0', 'conv3.1']):

                param.requires_grad = False

    return custom



def build_network_resnet18(pretrained = True, freeze_se = False):

    #Note, cannot load state dict for se_resnet18 

    if freeze_se == False:  

        resnet18 = models.resnet18(pretrained = pretrained)

        in_c = resnet18.fc.in_features

        resnet18.fc = nn.Linear(in_c, 2)

    else:

        #Create se_resnet18, customize classification layer

        resnet18 = se_resnet18(num_classes = 2)

        #Freeze first half of layers

        for name, param in resnet18.named_parameters():

            if not any(name.startswith(ext) for ext in ['layer3', 'layer4', 'fc']):

                param.requires_grad = False

    return resnet18



#Referenced works:

#https://github.com/pytorch/vision/blob/master/torchvision/models/googlenet.py

#https://github.com/moskomule/senet.pytorch/blob/master/senet/se_inception.py

def build_network_googlenet(pretrained = True, freeze_se = False):

    #Pretrain and customize classifier

    googlenet = models.googlenet(pretrained = pretrained)

    in_c = googlenet.fc.in_features

    googlenet.fc = nn.Linear(in_c, 2)

    

    #Add SELayers and freeze first half of layers

    if(freeze_se == True):

        for name, param in googlenet.named_parameters():

            if not any(name.startswith(ext) for ext in ['inception4c.branch3','inception4c.branch4', 'inception4d', 'inception4e', 'inception5', 'fc']):

                param.requires_grad = False

                

        #Add SELayers to each inception module when freeze_se is true

        googlenet.inception3a.add_module("SELayer", SELayer(32, 16))

        googlenet.inception3b.add_module("SELayer", SELayer(64, 16))

        googlenet.inception4a.add_module("SELayer", SELayer(64, 16))

        googlenet.inception4b.add_module("SELayer", SELayer(64, 16))

        googlenet.inception4c.add_module("SELayer", SELayer(64, 16))

        googlenet.inception4d.add_module("SELayer", SELayer(64, 16))

        googlenet.inception4e.add_module("SELayer", SELayer(128, 16))

        googlenet.inception5a.add_module("SELayer", SELayer(128, 16))

        googlenet.inception5b.add_module("SELayer", SELayer(128, 16))

        if googlenet.aux_logits:

            googlenet.aux1.add_module("SELayer", SELayer(2))

            googlenet.aux2.add_module("SELayer", SELayer(2))

    return googlenet



def build_network_resnet50(pretrained = True, freeze_se = False):

    if freeze_se == False:  

        resnet50 = models.resnet50(pretrained = pretrained)

        in_c = resnet50.fc.in_features

        resnet50.fc = nn.Linear(in_c, 2)

    else:

        #Note, although se_resnet50 can be pretrained, it's for 1K classes in the fc layer

        #We can only change our fc layer after we have pretrained

        resnet50 = se_resnet50(num_classes = 1000, pretrained = pretrained)

        #Change classification layer

        in_c = resnet50.fc.in_features

        resnet50.fc = nn.Linear(in_c, 2)

        #Freeze layers

        freeze_list = ['layer3.0.downsample', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5', 'layer4', 'fc']

        for name, param in resnet50.named_parameters():  

              if not any(name.startswith(ext) for ext in freeze_list):

                param.requires_grad = False

        

    return resnet50
summary(build_network_custom(),(3,96,96), device="cpu")

summary(build_network_custom(freeze_se = True),(3,96,96), device="cpu")

summary(build_network_resnet18(pretrained = True),(3,96,96), device="cpu")

summary(build_network_resnet18(pretrained = True, freeze_se = True),(3,96,96), device="cpu")

summary(build_network_googlenet(pretrained = True),(3,96,96), device="cpu")

summary(build_network_googlenet(pretrained = True,freeze_se = True),(3,96,96), device="cpu")

summary(build_network_resnet50(pretrained = True), (3,96,96), device="cpu")

summary(build_network_resnet50(pretrained = True, freeze_se = True),(3,96,96), device="cpu")
def train(net, epochs=5, lr=0.1, momentum=0.9):

    history_step=int(len(trainloader)/(150/epochs))

    train_loss_history = []

    

    # transfer model to GPU

    if torch.cuda.is_available():

       net = net.cuda()

    

    # set to training mode

    net.train()

    

    # train the network

    for e in notebook.tqdm(range(epochs),desc="Epoch"):   

        lr = lr/10

        total_loss = 0

        loss_count = 0



        for i, (inputs, labels) in notebook.tqdm(enumerate(trainloader),total=len(trainloader), desc="Epoch "+ str(e+1),leave=False):

              

            # set the optimizer. Use SGD with momentum

            optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=momentum)

            

            # Clear all the gradient to 0

            optimizer.zero_grad()



            # transfer data to GPU

            inputs = inputs.to(device)

            labels = labels.to(device)



            # forward propagation to get h

            outs = net(inputs)



            # compute loss 

            #change to loss = F.cross_entropy(outs.logits, labels) for se_googlenet if having problems 

            loss = F.cross_entropy(outs, labels)



            # backpropagation to get gradients of all parameters

            loss.backward()



            # update parameters

            optimizer.step()



            # get the loss

            total_loss += loss.item()

            loss_count += 1

            

            # display the averaged loss value 

            if i % history_step == 0 and i != 0:

                train_loss_history.append(total_loss/loss_count)

                total_loss=0

                loss_count=0



    return train_loss_history
custom = build_network_custom()

custom.name='custom'

hist_custom = train(custom)
se_custom = build_network_custom(freeze_se = True)

se_custom.name='se_custom'

hist_custom_se = train(se_custom)
resnet18 = build_network_resnet18(pretrained = True)

resnet18.name='resnet18'

hist_resnet18 = train(resnet18)
se_resnet18 = build_network_resnet18(pretrained = True, freeze_se = True)

se_resnet18.name='se_resnet18'

hist_se_resnet18 = train(se_resnet18)
googlenet = build_network_googlenet(pretrained = True)

googlenet.name='googlenet'

hist_googlenet = train(googlenet)
se_googlenet= build_network_googlenet(pretrained = True, freeze_se = True)

se_googlenet.name='se_googlenet'

hist_se_googlenet = train(se_googlenet)
resnet50 = build_network_resnet50(pretrained = True)

resnet50.name='resnet50'

hist_resnet50 = train(resnet50)
se_resnet50 = build_network_resnet50(pretrained = True, freeze_se = True)

se_resnet50.name='se_resnet50'

hist_se_resnet50= train(se_resnet50)
model_list = [custom, se_custom, resnet18, se_resnet18, googlenet, se_googlenet, resnet50, se_resnet50]
#test models without threshold, calc_plot_beta will calculate F1 and F2 scores for all thresholds

def test(model, threshold=None):

    model.eval()

    with torch.no_grad(): #set all the requires_grad flag to false

        pred = []

        test_y = []

        for images, labels in notebook.tqdm(testloader):

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            pred.append(outputs)

            test_y.append(labels)



    pred = torch.cat(pred, dim=0)#convert list to torch and flatten

    pred2 = pred.detach().cpu().numpy()

    pred_prob = pred2[:,1]#flatten and convert to array



    

    test_y = torch.cat(test_y, dim=0).cpu()

    if threshold==None:

        return pred_prob,test_y

    else:

        pred_binary = np.where(pred_prob > threshold,1,0)#set prediction become binary

        return pred_prob,pred_binary,test_y



def get_binary(pred_prob, threshold):

    pred_binary = np.where(pred_prob > threshold,1,0)#set prediction become binary

    return pred_binary
def plot_fbeta_threshold(beta, fbeta_scores, threshold, model_name_str):

    max_fbeta = np.sort(fbeta_scores)[len(fbeta_scores)-1]

    max_fbeta_idx = fbeta_scores.tolist().index(max_fbeta)

    xcoord = threshold[max_fbeta_idx]

    

   

    title1_str = model_name_str + " F-" + str(beta) +  " scores vs. Threshold"

    ylabel_str = "F-" + str(beta) + " scores"

    f_score_str = "F-" + str(beta) + " score vs. threshold"

    plt.figure()

    plt.title(title1_str)

    plt.plot(fbeta_scores,  'b', label =f_score_str)

    

    #plt.plot([0,1], [0,1], 'r--')

   

    plt.ylabel(ylabel_str)

    plt.xlabel('Threshold')

    threshold_label = "Threshold: " + str(np.round(xcoord,3))

    max_fbeta_label = "Max F-" + str(beta) + " score: " +  str(np.round(max_fbeta,3))

    plt.axvline(x = xcoord, c = 'r', label = threshold_label)

    plt.axhline(y = max_fbeta, c = 'y', label =max_fbeta_label )

  

    plt.legend(loc='lower right')

    plt.show()
from sklearn.metrics import fbeta_score



def calc_plot_fbeta(model, size):

    threshold = np.linspace(0,1, size)

    fbeta1_scores = []

    fbeta2_scores = []

    

    #For each threshold from 0 to 1, calculate the model's F1 & F2 score

    model_ans=test(model)

    for i in notebook.tqdm(threshold):

        y_pred_class = get_binary(model_ans[0], i)

        beta = 1  #calculate F1 score

        fbeta1 = fbeta_score(model_ans[1], y_pred_class, beta)

        fbeta1_scores = np.append(fbeta1_scores, fbeta1)

        beta = 2  #calculate F2 score

        fbeta2 = fbeta_score(model_ans[1], y_pred_class, beta)

        fbeta2_scores = np.append(fbeta2_scores, fbeta2)

    

    fbeta_scores = [(1, fbeta1_scores), (2, fbeta2_scores)]

    

    #Plot F1 vs. each threshold and F2 vs. each threshold for this model

    for i, fb_scores in fbeta_scores: 

        max_fbeta = np.sort(fb_scores)[len(fb_scores)-1]#get max score

        max_fbeta_idx = fb_scores.tolist().index(max_fbeta)# use max score to get current threshold

        xcoord = threshold[max_fbeta_idx]

        best_thresholds[str(model.name) + "_F" + str(i)] = xcoord, max_fbeta#record best thresholds and score

        plot_fbeta_threshold(i, fb_scores,  threshold, model.name)
best_thresholds = {} #store best thresholds (for F1 and F2 score) for each model



for architecture in model_list:

    #For each model in model_list, calculate F1 & F2 scores for various thresholds, and plot graph of scores vs. thresholds

    calc_plot_fbeta(architecture, 1000)
#Accuracy measurements

from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc

from sklearn.metrics import plot_confusion_matrix

import seaborn as sns



def show_acc_mea(result, model_name, t=0.5,plot=True):

    pred_prob=result[0]

    pred_binary=get_binary(pred_prob, t)

    test_y=result[1]

    

    #Print accuracy

    acc_NN = accuracy_score(test_y, pred_binary)

    print('Overall accuracy of Neural Network model:', acc_NN)



    #Print confusion matrix

    print('Confusion matrix:')

    cf_matrix = confusion_matrix(test_y, pred_binary)

    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

    

    if(plot):

        #Print Area Under Curve for Receiver Operating Characteristics AUROC/AUCROC

        false_positive_rate, recall, _ = roc_curve(test_y, pred_prob)

        roc_auc = auc(false_positive_rate, recall)

        plt.figure()

        plt.title(str(model_name) +  ', threshold = ' +  str(t) +  ' Receiver Operating Characteristic (ROC)')

        plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)

        plt.legend(loc='lower right')

        plt.plot([0,1], [0,1], 'r--')

        plt.xlim([0.0,1.0])

        plt.ylim([0.0,1.0])

        plt.ylabel('Recall/True-Positve rate')

        plt.xlabel('Fall-out (1-Specificity)/False-Positive rate')

    plt.show()



    #Comments

    #Higher AUC means better predictions





answer = {}



def print_acc_mea(net):

    print('----------------------------'+ net.name+'----------------------------------')

    answer[net.name]=test(net)#get prediction and y

    print("Thresholds: 0.5")

    show_acc_mea(answer[net.name], net.name) 

    for i in range(1, 3):

        t,s= best_thresholds[str(net.name)+ "_F" + str(i)]

        print("Best thresholds for F"+ str(i) +" :"+ str(t))

        print("Max F"+ str(i) +" score :"+ str(s))

        show_acc_mea(answer[net.name], net.name, t, False)



for architecture in model_list:

    print_acc_mea(architecture)
from sklearn.metrics import precision_recall_curve

from matplotlib import pyplot



def show_precision_recall(result, model_name):

    pred_prob=result[0]

    pred_binary=get_binary(pred_prob, 0.5)

    test_y=result[1]

    

    precision, recall, _ = precision_recall_curve(test_y, pred_binary)

    pr_curve_auc = auc(recall, precision)

    

    # plot the precision-recall curves

    no_skill = len(test_y[test_y==1]) / len(test_y)

    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill classifier')

    pyplot.plot(recall, precision, marker='.', label=model_name)

    # axis labels

    pyplot.xlabel('Recall')

    pyplot.ylabel('Precision')

    # show the legend

    pyplot.legend()

    # show the plot

    pyplot.show()
#Calculate precision recall for all classification thresholds 

for architecture in model_list:

    show_precision_recall(answer[architecture.name], architecture.name)

import matplotlib.pyplot as plt



training_history_list = [hist_custom, hist_custom_se, hist_resnet18, hist_se_resnet18, hist_googlenet, hist_se_googlenet, hist_resnet50, hist_se_resnet50]

j=0

for hist_training in training_history_list:

    plt.plot(hist_training, label= model_list[j].name)

    j+=1

plt.legend()

plt.show()

    

import os

import torch

import torchvision

import numpy as np

import pandas as pd

from PIL import Image

from torch import optim

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split



%matplotlib inline
DATASET_PATH = "../input/myntradataset/"

df = pd.read_csv(os.path.join(DATASET_PATH, "styles.csv"), error_bad_lines=False) # neglect lines with error
# Clean unexisting files

problems = []

for idx, line in df.iterrows():

    if not os.path.exists(os.path.join(DATASET_PATH, 'images', str(line.id)+'.jpg')):

        print(idx)

        problems.append(idx)

df.drop(df.index[problems], inplace=True)
# Split train and test dataset

train_df, test_df = train_test_split(df, test_size=0.2, random_state=2028)
train_df.head()
class FPIDataset(Dataset):

    """ Fashion Product Image Dataset

    """

    cat_list = df['masterCategory'].unique()

    cat2num = {cat:i for i, cat in enumerate(cat_list)}

    num2cat = {i:cat for i, cat in enumerate(cat_list)}

    def __init__(self, root, dataframe, transform=None):

        super(FPIDataset, self).__init__()

        self.dataframe = dataframe

        self.root = root

        if transform is None:

            transform = torchvision.transforms.Compose([

                torchvision.transforms.Resize((224, 224)),

                torchvision.transforms.ToTensor()

            ])

        self.transform = transform

        

    def __getitem__(self, idx):

        line = self.dataframe.iloc[idx]

        cat = line.masterCategory

        cat_id = self.cat2num[cat]

        img_path = os.path.join(self.root, str(line.id)+'.jpg')

        img = Image.open(img_path).convert('RGB')

        img_tensor = self.transform(img)

        return img_tensor, cat_id

            

    def __len__(self):

        return len(self.dataframe)
# Construct dataset and dataloader

train_ds = FPIDataset(os.path.join(DATASET_PATH, 'images'), train_df)

test_ds = FPIDataset(os.path.join(DATASET_PATH, 'images'), test_df)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
# Show cases

img_tensor, label = train_ds[0]

plt.imshow(img_tensor.numpy().transpose(1,2,0))

plt.title(FPIDataset.num2cat[label])

plt.axis('off')
from torchvision.models import ResNet, resnet18

import torch.nn as nn
class FPIModel(nn.Module):

    """ Fashion Product Image Model

    """

    def __init__(self, num_classes):

        super(FPIModel, self).__init__()

        backbone = resnet18(pretrained=True)

        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

        self.backbone = backbone

    def forward(self, inp):

        return self.backbone(inp)
model = FPIModel(len(df.masterCategory.unique()))
def train_val_loop(model, save_path):

    epochs = 1

    log_step = 20

    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    model.to(device)

    loss_hist = []

    for epoch in range(1, epochs+1):

        # train phase

        model.train()

        for bth_num, batch in enumerate(train_loader, 1):

            images, labels = batch

            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if bth_num % log_step == 0:

                print("\r Epoch: {}, # {} => Loss: {:.4f}".format(epoch, bth_num, loss.item()), flush=True, end='')

                loss_hist.append(loss.item())

        print()

        # Valid phase

        model.eval()

        total_preds = []

        total_labels = []

        for bth_num, batch in enumerate(test_loader, 1):

            images, labels = batch

            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            preds = logits.argmax(dim=-1)

            total_labels.append(labels.cpu().data.numpy())

            total_preds.append(preds.cpu().data.numpy())

        total_preds = np.concatenate(total_preds)

        total_labels = np.concatenate(total_labels)

        acc = sum(total_preds == total_labels) / len(total_labels)

        print("Accuracy: {:.2%}".format(acc))

    

    # Save model weights

    torch.save(model.state_dict(), save_path)

    

    return loss_hist
loss_hist = train_val_loop(model, 'resnet18.pth')
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,

                     padding=dilation, groups=groups, bias=False, dilation=dilation)
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
class SEBasicBlock(nn.Module):

    expansion = 1



    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,

                 base_width=64, dilation=1, norm_layer=None, reduction=16):

        super(SEBasicBlock, self).__init__()

        if norm_layer is None:

            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:

            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:

            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, 1)

        self.bn2 = norm_layer(planes)

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
def se_resnet18(num_classes=1_000):

    """Constructs a ResNet-18 model.

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    model.avgpool = nn.AdaptiveAvgPool2d(1)

    return model
se_model = se_resnet18(num_classes=len(df.masterCategory.unique()))
se_loss_hist = train_val_loop(se_model, 'se_resnet18.pth')
# Plot training process

plt.plot(loss_hist, label='resnet18')

plt.plot(se_loss_hist, label='se_resnet18')

plt.title('Train Loss History')

plt.xlabel('step')

plt.ylabel('loss')

plt.legend()

plt.savefig('loss_hist.png')
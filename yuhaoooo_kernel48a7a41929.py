# classing for loading dataset (including training dataset and testing dataset)
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MNIST(Dataset):
    
    def __init__(self, filepath, trans, mode='train'):
        super(Dataset, self).__init__()
        self.file = pd.read_csv(filepath)
        self.trans = trans
        self.mode = mode
    
    def __getitem__(self, index):
        line = self.file.loc[index].to_numpy()
        if self.mode == 'train':
            label, image = line[0], line[1:].astype(np.uint8).reshape(28, 28)
        else:
            image = line[0:].astype(np.uint8).reshape(28, 28)
        image = Image.fromarray(image)
        if self.trans is not None:
            transed_image = self.trans(image)
            if self.mode == 'train':
                return label, transed_image
            return transed_image
        if self.mode == 'train':
            return label, image
        return image
            

    def __len__(self):
        return self.file.shape[0]
        
# Network
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride=2, norm=True):
        super(ConvBlock, self).__init__()
        self.norm = norm
        if norm is True:
            self.bn = nn.BatchNorm2d(input_channels)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.down = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.norm is True:
            input = self.bn(input)
        x = self.conv(input)
        x = self.down(x)
        x = self.relu(x)
        return x

class SpatialTransformerNetwork(nn.Module):
    
    def __init__(self, in_channels, img_size):
        super(SpatialTransformerNetwork, self).__init__()
        self.conv = nn.Sequential(*[
            ConvBlock(in_channels, in_channels * 2, 2, norm=False),
            ConvBlock(in_channels * 2, in_channels * 4, 2, norm=False)
        ])
        self.linear = nn.Linear(img_size * img_size * in_channels // 4, 6)

    def forward(self, input):
        batch_size = input.shape[0]
        x = self.conv(input)
        x = x.reshape(batch_size, -1)
        theta = self.linear(x)
        theta = theta.reshape(-1, 2, 3)
        grid = F.affine_grid(theta, input.size(), align_corners=True)
        x = F.grid_sample(input, grid, mode='bilinear', align_corners=True)
        return x
    

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # stn
        self.stn = SpatialTransformerNetwork(1, 32)
        self.blocks = nn.Sequential(*[
            ConvBlock(1, 64),    # 64 * 16 * 16
            ConvBlock(64, 128),  # 128 * 8 * 8
            ConvBlock(128, 256), # 256 * 4 * 4
        ])
        self.linear = nn.Linear(256 * 4 * 4, 10)

    def forward(self, input):
        batch_size = input.shape[0]
        input = F.interpolate(input, (32, 32), mode='bilinear', align_corners=True)
        x = self.stn(input)
        x = self.blocks(x)
        x = x.reshape(batch_size, -1)
        return self.linear(x)
# Hyper parameters
batch_size = 64
shuffle = True
epochs = 40
sample_interval = 50
lr = 5e-4
# loading training dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
trans = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomRotation(10),
    transforms.RandomCrop(28),
    transforms.ToTensor(),
    transforms.Normalize([0.], [255.0])
])
train_dataset = MNIST('/kaggle/input/digit-recognizer/train.csv', trans, 'train')
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

network = Network().cuda()
optimizer = optim.Adam(network.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

loss_list = []
acc_list = []
step = 0
# training network
for epoch in range(epochs):
    
    for idx, batch in enumerate(loader):
        label, image = batch[0].cuda(), batch[1].cuda()
        
        optimizer.zero_grad()
        logit = network(image)
        # acc
        predict = torch.argmax(logit, dim=1)
        acc = (predict == label).float().mean()

        loss = loss_fn(logit, label)
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % sample_interval == 0:
            loss_list += [loss.detach().cpu().item()]
            acc_list += [acc.detach().cpu().item()]

torch.save(network, '/kaggle/working/recog_{}.pth'.format(epochs))
plt.figure()
plt.plot(range(len(loss_list)), loss_list, 'b-', label='loss')
plt.plot(range(len(acc_list)), acc_list, 'r-', label='accuracy')
plt.legend()

# testing dataset
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.], [255.0])
])
testing_dataset = MNIST('/kaggle/input/digit-recognizer/test.csv', trans, 'test')
len_testing_dataset = len(testing_dataset)
loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
network.eval()
results = []
for idx, batch in enumerate(loader):
    batch_size = batch.shape[0]
    batch = batch.cuda()
    logit = network(batch)
    predict = torch.argmax(logit, dim=1)
    predict = predict.detach().cpu().numpy()
    results += [predict.reshape(batch_size, 1)]
df = np.concatenate(results, axis=0)
index = np.arange(len_testing_dataset).reshape(len_testing_dataset, 1) + 1
df = np.concatenate([index, df], axis=1)

# saving predictions
frame = pd.DataFrame(data=df)
frame.to_csv('/kaggle/working/submission.csv', header=['ImageId', 'Label'], index=False)
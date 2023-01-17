# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
class MNISTDataset(Dataset):

    def __init__(self, csv_file, root_dir, data_type, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.digit_frame = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.digit_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_type+'_'+str(self.digit_frame.iloc[idx, 0])+'.jpg')
        image = io.imread(img_name)
        image = image.reshape(image.shape[0], image.shape[1],1)
        digit = self.digit_frame.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, digit
'''class ToTensor(object):
    def __call__(self, sample):
        image, digit = sample['image'], sample['digit']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(np.reshape(image,(1,image.shape[0],image.shape[1]))).to(torch.float),
                'digit': torch.tensor(digit, dtype=torch.long)}
class Normalize(transforms.Normalize):
    def __call__(self, sample):
        return super(Normalize, self).__call__(sample)'''
'''fig = plt.figure()

def show_image(image):
    plt.imshow(image.reshape(image.size(1),image.size(2)))
    plt.pause(0.001)  # pause a bit so that plots are updated

datasets = {}
dataloaders = {}
for data_type in ('train', 'valid'):
    csv_file = os.path.join('..', 'input', data_type+'_label.csv')
    root_dir = os.path.join('..', 'input', data_type, data_type)
    dataset = MNISTDataset(csv_file=csv_file,
                                 root_dir=root_dir,
                                 data_type=data_type,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    datasets[data_type]=dataset
    dataloaders[data_type]=dataloader
    for i in range(len(dataset)):
        image, digit = dataset[i]

        print(i, digit)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_image(image)

        if i == 3:
            plt.show()
            break'''
'''# Helper function to show a batch
def show_images_batch(images_batch):
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    
    grid = utils.make_grid(images_batch.reshape((batch_size,1,im_size,im_size)))
    plt.imshow(grid.numpy()[0])

for i_batch, (images_batch, digits_batch) in enumerate(dataloader):
    print(i_batch, images_batch.size(), digits_batch.size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_images_batch(images_batch)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv1_bn = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_bn(self.fc2(x))
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch, \
          train_losses, train_corrects, maxes_train_correct, epochs_argmax_train_correct):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        train_loss += F.nll_loss(output, target, reduction='sum').item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_corrects.append(correct)
    
    if maxes_train_correct[-1]<correct:
        maxes_train_correct.append(correct)
        epochs_argmax_train_correct.append(epoch)
    
    print('\nTrain set: Average loss: {:.6f}, Accuracy: {}/{} ({:.3f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    print('Train set: Max Accuracy: {}/{} ({:.3f}%), Epoch: {}'.format(
        maxes_train_correct[-1], len(train_loader.dataset),
        100. * maxes_train_correct[-1] / len(train_loader.dataset), epochs_argmax_train_correct[-1]))
            
def test(args, model, device, valid_loader, test_loader, epoch, \
          valid_losses, valid_corrects, maxes_valid_correct, epochs_argmax_valid_correct, submission_np):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    valid_loss /= len(valid_loader.dataset)
    valid_losses.append(valid_loss)
    valid_corrects.append(correct)
    
    if maxes_valid_correct[-1]<correct:
        maxes_valid_correct.append(correct)
        epochs_argmax_valid_correct.append(epoch)
        
        submission_np[0] = np.empty(shape=[0, 2])
        batch_size = round(len(test_loader.dataset)/len(test_loader))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                sample_idx=batch_idx*batch_size
                idx = np.arange(sample_idx, sample_idx+target.size(0))
                new_submission_np = np.hstack((idx.reshape(-1,1), pred.cpu().numpy()))
                submission_np[0]=np.concatenate((submission_np[0], new_submission_np), axis=0)

    print('\nValid set: Average loss: {:.6f}, Accuracy: {}/{} ({:.3f}%)'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    print('Valid set: Max Accuracy: {}/{} ({:.3f}%), Epoch: {}\n'.format(
        maxes_valid_correct[-1], len(valid_loader.dataset),
        100. * maxes_valid_correct[-1] / len(valid_loader.dataset), epochs_argmax_valid_correct[-1]))

class args_t:
    def __init__(self):
        self.batch_size=64
        self.test_batch_size=1000
        self.epochs=100
        self.lr=0.01
        self.momentum=0.5
        self.no_cuda=False
        self.seed=1
        self.log_interval=10
        self.save_model=False

mnistdatasets = {}
dataloaders = {}
        
def plot2(train_losses, valid_losses, train_corrects, valid_corrects):
    plt.title("Train and Valid Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    train_sample=len(dataloaders['train'].dataset)
    valid_sample=len(dataloaders['valid'].dataset)

    plt.plot(range(1,len(train_losses)+1),[100.*loss/train_sample for loss in train_losses],label="Train set")
    plt.plot(range(1,len(valid_losses)+1),[100.*loss/valid_sample for loss in valid_losses],label="Valid set")
    plt.legend()
    plt.show()

    plt.title("Train and Valid Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    train_sample=len(dataloaders['train'].dataset)
    valid_sample=len(dataloaders['valid'].dataset)

    plt.plot(range(1,len(train_corrects)+1),[100.*correct/train_sample for correct in train_corrects],label="Train set")
    plt.plot(range(1,len(valid_corrects)+1),[100.*correct/valid_sample for correct in valid_corrects],label="Valid set")
    plt.ylim((100.*valid_corrects[0]/valid_sample),100)
    plt.legend()
    plt.show()

Nets = {}
lrs = {}

losses = {}
corrects = {}
maxes_correct = {}
epochs_argmax_correct= {}
submission_np = {}
        
def main():
    # Training settings
    args = args_t()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    for data_type in ('train', 'valid', 'test'):
        if data_type=='test':
            csv_file_name = 'valid_label.csv'
        else:
            csv_file_name = data_type+'_label.csv'
        csv_file = os.path.join('..', 'input', csv_file_name)
        root_dir = os.path.join('..', 'input', data_type, data_type)
        dataset = MNISTDataset(csv_file=csv_file,
                                     root_dir=root_dir,
                                     data_type=data_type,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))
                                     ]))
        if data_type=='test':
            digit_frame = dataset.digit_frame
            digit_frame.iloc[:,1] = np.arange(digit_frame.shape[0])
            shuffle=False
        else:
            shuffle=True
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=shuffle, **kwargs)
        mnistdatasets[data_type]=dataset
        dataloaders[data_type]=dataloader
        
    train_loader=dataloaders['train']
    valid_loader=dataloaders['valid']
    test_loader=dataloaders['test']
    
    Nets['0']=Net1()
    Nets['1']=Net2()
    
    lrs['0']=0.01
    lrs['1']=0.1
    for i in range(2):
        i=str(i)
        
        device = torch.device("cuda" if use_cuda else "cpu")
        model = Nets[i].to(device)
        optimizer = optim.SGD(model.parameters(), lr=lrs[i], momentum=args.momentum)
        
        for data_type in ('train', 'valid'):
            key=i+data_type
            losses[key]=[]
            corrects[key]=[]
            maxes_correct[key]=[0]
            epochs_argmax_correct[key]=[]
        submission_np[i]=[0]
        
        for epoch in range(1, args.epochs + 1):
            key=i+'train'
            train(args, model, device, train_loader, optimizer, epoch, \
                  losses[key], corrects[key], \
                  maxes_correct[key], epochs_argmax_correct[key])
            key=i+'valid'
            test(args, model, device, valid_loader, test_loader, epoch, \
                  losses[key], corrects[key], \
                  maxes_correct[key], epochs_argmax_correct[key], submission_np[i])

        maxes_correct[i+'train'].pop(0)
        maxes_correct[i+'valid'].pop(0)
        np.savetxt('submission'+i+'.csv', submission_np[i][0], fmt='%d', delimiter=',', header='id,label', comments='')
        
        plot2(losses[i+'train'], losses[i+'valid'], corrects[i+'train'], corrects[i+'valid'])
        
        if (args.save_model):
            torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, index, title = "Download CSV file"):  
    csv = df.to_csv(index=index)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
filename='submission0.csv'
df=pd.read_csv(filename, header=0)
create_download_link(df, index=False)
# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
filename='submission1.csv'
df=pd.read_csv(filename, header=0)
create_download_link(df, index=False)





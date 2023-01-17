import numpy as np
import pandas as pd
from PIL import Image
#import seaborn as sns
#import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import StratifiedKFold
class ImageDataset(Dataset):
    def __init__(self, csv_file, img_path, transform=None):

        self.csv_file = csv_file
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        #csv_file=train_labels
        #img_path="D:/Human Protein Atlas Image Classification/train/"
        #idx=0
        #path= img_path+csv_file.iloc[idx, 0]
        
        path = self.img_path+self.csv_file.iloc[idx, 0]
        
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R)/255, 
            np.array(G)/255, 
            np.array(B)/255,
            np.array(Y)/255))
        
        im=torch.Tensor(im)
        label = torch.from_numpy(np.array(list(self.csv_file.iloc[idx,1:]))).float()
        
        return im, label
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.C1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3,padding=1)
        self.C2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3,padding=1)
        
        self.C3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,padding=1)
        self.C4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,padding=1)
        
        self.C5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1)
        self.C6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1)
        self.C7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1)
        self.C8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1)
        
        self.C9 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1)
        self.C10 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        self.C11 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        self.C12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)

        self.L1 = nn.Linear(32*32*64, 512)
        self.L2 = nn.Linear(512, 28)

    def forward(self, x):
        x=self.C1(x)
        x=self.C2(x)
        x=F.max_pool2d(F.relu(x),2)

        x=self.C3(x)
        x=self.C4(x)
        x=F.max_pool2d(F.relu(x),2)
        
        x=self.C5(x)
        x=self.C6(x)
        x=self.C7(x)
        x=self.C8(x)
        x=F.max_pool2d(F.relu(x),2)
        
        x=self.C9(x)
        x=self.C10(x)
        x=self.C11(x)
        x=self.C12(x)
        x=F.max_pool2d(F.relu(x),2)
        
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        
        x=F.sigmoid(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
def FocalLoss(output, target):
    gamma=2
    if not (target.size() == output.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), output.size()))

    max_val = (-output).clamp(min=0)
    loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

    invprobs = F.logsigmoid(-output * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss
        
    return loss.sum(dim=1).mean()

if __name__ == "__main__":
    train_labels = pd.read_csv("D:/Human Protein Atlas Image Classification/train.csv")
    
    train_labels.head()
    train_labels.shape[0]

    label_names = {
        0:  "Nucleoplasm",  
        1:  "Nuclear membrane",   
        2:  "Nucleoli",   
        3:  "Nucleoli fibrillar center",   
        4:  "Nuclear speckles",
        5:  "Nuclear bodies",   
        6:  "Endoplasmic reticulum",   
        7:  "Golgi apparatus",   
        8:  "Peroxisomes",   
        9:  "Endosomes",   
        10:  "Lysosomes",   
        11:  "Intermediate filaments",   
        12:  "Actin filaments",   
        13:  "Focal adhesion sites",   
        14:  "Microtubules",   
        15:  "Microtubule ends",   
        16:  "Cytokinetic bridge",   
        17:  "Mitotic spindle",   
        18:  "Microtubule organizing center",   
        19:  "Centrosome",   
        20:  "Lipid droplets",   
        21:  "Plasma membrane",   
        22:  "Cell junctions",   
        23:  "Mitochondria",   
        24:  "Aggresome",   
        25:  "Cytosol",   
        26:  "Cytoplasmic bodies",   
        27:  "Rods & rings"
    }

    reverse_train_labels = dict((v,k) for k,v in label_names.items())

    for key in label_names.keys():
        train_labels[label_names[key]] = 0
        
    train_labels = train_labels.apply(fill_targets, axis=1)
    train_labels.head()
    del train_labels['Target']
    
#check sample is ok
#000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue

    transformations = transforms.Compose([transforms.ToTensor()])
    
    model = Net()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.99))

    for k in train_labels.columns:
        if k!='Id':
            folds = StratifiedKFold(n_splits=10, shuffle=True)
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_labels.values,train_labels[k].values)):
                
                temp=train_labels.iloc[val_idx].reset_index(drop=True)

                dataset = ImageDataset(csv_file=temp,img_path="D:/Human Protein Atlas Image Classification/train/",transform=transformations )
                
                dataloader = DataLoader(dataset, batch_size=64,shuffle=True)
                for batch_idx, (data, target) in enumerate(dataloader):
                    print('1')

                    data, target = Variable(data), Variable(target)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = FocalLoss(output, target)
                    loss.backward()
                    optimizer.step()
                    print('Train Epoch: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(data), len(dataloader.dataset),100. * batch_idx / len(dataloader), loss.data[0]))

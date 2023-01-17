import os

import torch

import pandas as pd

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from PIL import Image

import torchvision.models as models

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from sklearn.metrics import f1_score

import torch.nn.functional as F

import torch.nn as nn

from torchvision.utils import make_grid

%matplotlib inline
DATA_DIR = '../input/jovian-pytorch-z2g/Human protein atlas'



TRAIN_DIR = DATA_DIR + '/train'                           # Contains training images

TEST_DIR = DATA_DIR + '/test'                             # Contains test images



TRAIN_CSV = DATA_DIR + '/train.csv'                       # Contains real labels for training images

TEST_CSV = '../input/jovian-pytorch-z2g/submission.csv'   # Contains dummy labels for test image
train_df = pd.read_csv(TRAIN_CSV)

labels = {

    0: 'Mitochondria',

    1: 'Nuclear bodies',

    2: 'Nucleoli',

    3: 'Golgi apparatus',

    4: 'Nucleoplasm',

    5: 'Nucleoli fibrillar center',

    6: 'Cytosol',

    7: 'Plasma membrane',

    8: 'Centrosome',

    9: 'Nuclear speckles'

}
def encode_label(label):

    #create an initial target which is all zeros

    target = torch.zeros(10)

    for l in str(label).split(' '):

        target[int(l)] = 1.

    return target



def decode_target(target, text_labels=False, threshold=0.5):

    result = []

    for i, x in enumerate(target):

        if (x >= threshold):

            if text_labels:

                result.append(labels[i] + "(" + str(i) + ")")

            else:

                result.append(str(i))

    return ' '.join(result)



def show_sample(img, target, invert=True):

    if invert:

        plt.imshow(1 - img.permute((1, 2, 0)))

    else:

        plt.imshow(img.permute(1, 2, 0))

    print('Labels:', decode_target(target, text_labels=True))

    

def show_batch(dl, invert=True):

    for images, labels in dl:

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.set_xticks([]); ax.set_yticks([])

        data = 1-images if invert else images

        ax.imshow(make_grid(data, nrow=16).permute(1, 2, 0))

        break

        

def get_default_device():

    """Pick GPU if available, else CPU"""

    if torch.cuda.is_available():

        return torch.device('cuda')

    else:

        return torch.device('cpu')

    

def to_device(data, device):

    """Move tensor(s) to chosen device"""

    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)
class HumanProteinDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        self.df = pd.read_csv(csv_file)

        self.transform = transform

        self.root_dir = root_dir

        

    def __len__(self):

        return len(self.df)    

    

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img_id, img_label = row['Image'], row['Label']

        img_fname = self.root_dir + "/" + str(img_id) + ".png"

        img = Image.open(img_fname)

        if self.transform:

            img = self.transform(img)

        return img, encode_label(img_label)
transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, transform=transform)
torch.manual_seed(10)

val_pct = 0.1

val_size = int(val_pct * len(dataset))

train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

len(train_ds), len(val_ds)
batch_size = 64

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)
show_batch(train_dl)
def F_score(output, label, threshold=0.5, beta=1):

    prob = output > threshold

    label = label > threshold



    TP = (prob & label).sum(1).float()

    TN = ((~prob) & (~label)).sum(1).float()

    FP = (prob & (~label)).sum(1).float()

    FN = ((~prob) & label).sum(1).float()



    precision = torch.mean(TP / (TP + FP + 1e-12))

    recall = torch.mean(TP / (TP + FN + 1e-12))

    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return F2.mean(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg16 = models.vgg16(pretrained=True)

vgg16 
for param in vgg16.parameters():

  param.require_grad = False
fc = nn.Sequential(

    nn.Linear(25088, 5000),

    nn.ReLU(),

    nn.Dropout(0.5),

    

    nn.Linear(5000, 1000),

    nn.ReLU(),

    nn.Dropout(0.5),

    

    nn.Linear(1000, 460),

    nn.ReLU(),

    nn.Dropout(0.5),

    

    nn.Linear(460,10),

    

)



vgg16.classifier = fc

vgg16
class MultilabelImageClassificationBase(nn.Module):

    def training_step(self, batch):

        images, targets = batch 

        out = self(images)                      

        loss = F.binary_cross_entropy(out, targets)      

        return loss

    

    def validation_step(self, batch):

        images, targets = batch 

        out = self(images)                           # Generate predictions

        loss = F.binary_cross_entropy(out, targets)  # Calculate loss

        score = F_score(out, targets)

        return {'val_loss': loss.detach(), 'val_score': score.detach() }

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        batch_scores = [x['val_score'] for x in outputs]

        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    

    def epoch_end(self, epoch, result):

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(

            epoch, result['train_loss'], result['val_loss'], result['val_score']))
class DeviceDataLoader():

    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):

        self.dl = dl

        self.device = device

        

    def __iter__(self):

        """Yield a batch of data after moving it to device"""

        for b in self.dl: 

            yield to_device(b, self.device)



    def __len__(self):

        """Number of batches"""

        return len(self.dl)
class TransferModel(MultilabelImageClassificationBase):

    def __init__(self, model):

        super().__init__()

        # Use a pretrained model

        self.network = model

    

    def forward(self, xb):

        return torch.sigmoid(self.network(xb))
device = get_default_device()

device
train_dl = DeviceDataLoader(train_dl, device)

val_dl = DeviceDataLoader(val_dl, device)
from tqdm.notebook import tqdm

@torch.no_grad()

def evaluate(model, val_loader):

    model.eval()

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):

    torch.cuda.empty_cache()

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        model.train()

        train_losses = []

        for batch in tqdm(train_loader):

            loss = model.training_step(batch)

            train_losses.append(loss)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        result['train_loss'] = torch.stack(train_losses).mean().item()

        model.epoch_end(epoch, result)

        history.append(result)

    return history
model = to_device(TransferModel(vgg16), device)
num_epochs = 1

opt_func = torch.optim.Adam

lr = .0001

fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
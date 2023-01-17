!pip install timm
import os

import numpy as np 

import pandas as pd 

import glob

import openslide

import PIL

import torch

import tqdm

import timm

from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cpu = torch.device('cpu')

print('Running on device: {}'.format(device))
train_img_dir = '../input/prostate-cancer-grade-assessment/train_images/'

train_msk_dir = '../input/prostate-cancer-grade-assessment/train_label_masks/'
train_meta = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

test_meta = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

train_img_dir = '../input/prostate-cancer-grade-assessment/train_images/'

train_mask_dir = '../input/prostate-cancer-grade-assessment/train_label_masks/'

# sample = pd.read_csv('../input/prostate-cancer-grade-assessment/sample_submission.csv')
from PIL import Image, ImageChops, ImageOps

from sklearn.feature_extraction.image import extract_patches_2d



def trim(im):

    bg = Image.new(im.mode, im.size, (255,255,255))

    diff = ImageChops.difference(im, bg)

    diff = ImageChops.add(diff, diff, 2.0, -100)

    bbox = diff.getbbox()

    if bbox:

        return im.crop(bbox)

def tile(img, N):

    sz = 256

    result = []

    shape = img.shape

    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz

    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],

                constant_values=255)

    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)

    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)

    if len(img) < N:

        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]

    img = img[idxs]

    return img



def pre_process(image_id, n_patches= 10):

    """Show a mask overlayed on a slide."""



    slide = openslide.OpenSlide(os.path.join(train_img_dir, image_id + '.tiff'))

    slide_data = slide.read_region((0,0), 2, slide.level_dimensions[2])

    slide_data_crop = trim(Image.fromarray(np.asarray(slide_data)[:,:,0:3], 'RGB'))

    if slide_data_crop == None:

        slide_data_crop = Image.fromarray(np.asarray(slide_data)[:,:,0:3], 'RGB')

    w, h = slide_data_crop.size

    if (h < 266) or (w < 266):

        slide_data_crop = slide_data_crop.resize(size = (np.max([w, 266]), np.max([h, 266])))

    image_crop = np.asarray(slide_data_crop)

#     image_patches = extract_patches_2d(image_crop, (256,256), max_patches=n_patches, random_state=None)

    image_patches = tile(image_crop, N = 10)

    image_patches = np.transpose(image_patches, (0, 3, 1, 2)).astype(np.float32)



    slide.close()

    return(image_patches)
image_o = pre_process(image_id = 'a3794ec31a02fbf429486dd464f83d25')

image_o.shape
from torch.utils import data

class Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, labels):

        'Initialization'

        self.labels = labels

        self.list_IDs = list_IDs



    def __len__(self):

        'Denotes the total number of samples'

        return len(self.list_IDs)



    def __getitem__(self, index):

        'Generates one sample of data'

        # Select sample

        ID = self.list_IDs[index]

        # Load data and get label

#         print(index)

        X = pre_process(image_id = ID)

        y = self.labels[index]



        return X, y
dataset = Dataset(list_IDs = train_meta.image_id,

                  labels = train_meta.isup_grade)

train_data, val_data = torch.utils.data.random_split(dataset, [8000, 2616])
train_batch_size = 5

val_batch_size = 5

train_loader = DataLoader(train_data, shuffle=False, batch_size=train_batch_size, num_workers = 4)

val_loader = DataLoader(val_data, shuffle=False, batch_size=val_batch_size, num_workers = 4)
# mixnet = timm.create_model("mixnet_s", pretrained=True)

# del(mixnet)
# count = 0

# for inputs, labels in train_loader:

#     if count > 0:

#         break

#     print(inputs.shape)

#     count += 1

# mixnet.to(device)
import torch.nn as nn



def qwk3(a1, a2, max_rat=6):

    assert(len(a1) == len(a2))

    a1 = np.asarray(a1, dtype=np.int32)

    a2 = np.asarray(a2, dtype=np.int32)



    hist1 = np.zeros((max_rat + 1, ))

    hist2 = np.zeros((max_rat + 1, ))



    o = 0

    for k in range(a1.shape[0]):

        i, j = a1[k], a2[k]

        hist1[i] += 1

        hist2[j] += 1

        o +=  (i - j) * (i - j)



    e = 0

    for i in range(max_rat + 1):

        for j in range(max_rat + 1):

            e += hist1[i] * hist2[j] * (i - j) * (i - j)



    e = e / a1.shape[0]



    return 1 - o / e



class Identity(nn.Module):

    def __init__(self):

        super(Identity, self).__init__()

        

    def forward(self, x):

        return x

    

class PandaNet(nn.Module):

    def __init__(self, drop_prob=0.5):

        super(PandaNet, self).__init__()

                

        self.mixnet = timm.create_model("mixnet_s", pretrained=True)

        self.mixnet.classifier = Identity()

        self.lstm = nn.LSTM(1536, 256, 2, dropout=.75, bidirectional=True, batch_first=True)

        self.fc1 = nn.Linear(512, 128)

        self.fc4 = nn.Linear(128, output_size)

        self.elu = nn.ELU()

        self.sigmoid = nn.Sigmoid()

        

    def forward(self, x):

        b, p, c, h, w = x.shape

        out = self.mixnet(x.reshape(b*p, c, h, w))

        out = out.reshape(b, p, 1536)

        out, _ = self.lstm(out)

        out, _ = torch.max(out, 1)

        out = self.fc1(out)

        out = self.elu(out)

        out = self.fc4(out)

        out = self.sigmoid(out)

        

        out = out.view(x.size()[0], -1)

#         out = out[:,-1]

        return out
input_size = 512

output_size = 6

hidden_dim = 512



model = PandaNet()

model.to(device)
train_criterion = nn.CrossEntropyLoss()

val_criterion = nn.CrossEntropyLoss()



optimizer = optim.Adam(model.parameters(), lr=0.0001)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
epochs = 10

counter = 0

print_every = 800

clip = .5

valid_loss_min = np.Inf

val_loss = torch.tensor(np.Inf)

model.train()

for i in range(epochs):

    for inputs, labels in train_loader:

        counter += 1

        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()

        output = model(inputs)

        loss = train_criterion(output, labels)

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        loss.backward()

        optimizer.step()

        train_qwk = qwk3(output.argmax(1).detach().cpu(), labels.cpu())

        torch.cuda.empty_cache() 

        if counter % 100 == 0:

            print("Time: {}...".format(datetime.now().strftime("%H:%M:%S")) + 

                  "Epoch: {}/{}...".format(i+1, epochs) +  

                  "Step: {}...".format(counter) +

                  "Loss: {:.6f}...".format(loss.item())) 

        

        if counter%print_every == 0:

            val_losses = []

            val_qwk = []

            model.eval()

            for inp, lab in val_loader:

                

                inp, lab = inp.to(device), lab.to(device)

                out = model(inp)

                val_loss = val_criterion(out, lab)

                val_losses.append(val_loss.item())

                val_qwk.append(qwk3(out.argmax(1).detach().cpu(), lab.cpu()))

            model.train()

            print("Time: {}...".format(datetime.now().strftime("%H:%M:%S")) + 

                  "Epoch: {}/{}...".format(i+1, epochs),

                  "Step: {}...".format(counter),

                  "Loss: {:.6f}...".format(loss.item()),

                  "Val Loss: {:.6f}".format(np.mean(val_losses)),

                  "QWK: {:.2f}...".format(np.mean(val_qwk)))

            if np.mean(val_losses) <= valid_loss_min:

                torch.save(model.state_dict(), './model_mz.pt')

                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))

                valid_loss_min = np.mean(val_losses)

                torch.cuda.empty_cache() 

    scheduler.step(loss.item())
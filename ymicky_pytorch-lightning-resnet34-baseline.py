# install pytorch-lightning

!pip install pytorch-lightning
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import os

import random as rn



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from PIL import Image, ImageFile



import torch

from torch.nn import functional as F

import torch.nn as nn

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from torchvision import transforms as T



import pytorch_lightning as pl

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import ModelCheckpoint

#fix random seed

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(42)

rn.seed(12345)

torch.manual_seed(2019)

torch.cuda.manual_seed(2019)

torch.cuda.manual_seed_all(2019)

torch.backends.cudnn.deterministic = True
# ref: https://www.kaggle.com/yhn112/resnet18-baseline-pytorch-ignite

class RCICDataset(Dataset):

    def __init__(

        self,

        df,

        img_dir,

        mode='train',

        site=1,

        debug=False

        ):



        self.df = df

        if debug:

            self.df = df[:100]

        self.records = self.df.to_records(index=False)

        self.channels = [1,2,3,4,5,6]

        self.site = site

        self.mode = mode

        self.debug = debug

        self.img_dir = str(img_dir)

        self.len = self.df.shape[0]

        self.size = 256



    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)

#             return np.array(img)



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,

                         self.mode,

                         experiment,

                         'Plate{}'.format(plate),

                         '{}_s{}_w{}.png'.format(well,

                                                 self.site,

                                                 channel)])



    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        img = torch.cat([self._load_img_as_tensor(img_path)

                         for img_path in paths])

        if self.mode == 'train':

            return img, self.records[index].sirna

        else:

            return img, self.records[index].id_code



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len
# define variables

epoch = 10

n_class = 1108

DATA_DIR = '../input/recursion-cellular-image-classification'

debug = False

batchsize = 64

num_workers = 4
class RCICSystem(pl.LightningModule):



    def __init__(self, train_loader, val_loader, model):

        super(RCICSystem, self).__init__()

        # not the best model...

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.model = model

        self.criteria = nn.CrossEntropyLoss()



    def forward(self, x):

        return self.model(x)



    def training_step(self, batch, batch_nb):

        # REQUIRED

        x, y = batch

        y_hat = self.forward(x)

        loss = self.criteria(y_hat, y)

        loss = loss.unsqueeze(dim=-1)

        return {'loss': loss}



    def validation_step(self, batch, batch_nb):

        # OPTIONAL

        x, y = batch

        y_hat = self.forward(x)

        val_loss = self.criteria(y_hat, y)

        val_loss = val_loss.unsqueeze(dim=-1)

        return {'val_loss': val_loss}



    def validation_end(self, outputs):

        # OPTIONAL

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        return {'avg_val_loss': avg_loss}



    def configure_optimizers(self):

        # REQUIRED

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        return [optimizer], [scheduler]



    @pl.data_loader

    def tng_dataloader(self):

        # REQUIRED

        return self.train_loader



    @pl.data_loader

    def val_dataloader(self):

        # OPTIONAL

        return self.val_loader

    @pl.data_loader

    def test_dataloader(self):

        # OPTIONAL

        pass



train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

train_df, val_df = train_test_split(train_df, stratify=train_df['sirna'])
# you can also define a checkpoint callback to save best model like keras.

checkpoint_callback = ModelCheckpoint(

    filepath='../working',

    save_best_only=True,

    verbose=True,

    monitor='avg_val_loss',

    mode='min'

)
# get resnet34 model with 6 channels

from torchvision.models import resnet34

def get_model(pretrained=False):

    model = resnet34(pretrained=pretrained)

    model.fc = nn.Linear(512, n_class)

    trained_kernel = model.conv1.weight

    new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():

        new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

    model.conv1 = new_conv

    return model
train = RCICDataset(train_df, DATA_DIR, debug=debug)

train_loader = DataLoader(train, batch_size=batchsize, pin_memory=True,

                          shuffle=True)

val = RCICDataset(val_df, DATA_DIR)

val_loader = DataLoader(val, batch_size=batchsize, pin_memory=True,

                        shuffle=False)
model = get_model()


pl_model = RCICSystem(train_loader, val_loader, model)



# set gpus, epoch and callbacks

trainer = Trainer(gpus=[0], max_nb_epochs=epoch,

                  checkpoint_callback=checkpoint_callback)

# fit model !

trainer.fit(pl_model)
os.listdir('../working/')
from collections import OrderedDict

def load_pytorch_model(state_dict, *args, **kwargs):

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():

        name = k

        if name.startswith('model.'):

            name = name.replace('model.', '') # remove `model.`

        new_state_dict[name] = v

    model = get_model(pretrained=False)

    model.load_state_dict(new_state_dict)

    return model
# load best model

from pathlib import Path

ckpt_path = list(Path('../working').glob('*.ckpt'))[0]

ckpt_dict = torch.load(ckpt_path)

best_model = load_pytorch_model(ckpt_dict['state_dict'])
def predict(model, dataloader, n_class, device, tta=1):

    model.eval()

    model.to(device)

    preds = np.zeros([0, n_class])

    for data, _ in dataloader:

        data = data.to(device)

        with torch.no_grad():

            y_pred = model(data).detach()

        #y_pred = F.softmax(y_pred, dim=1).cpu().numpy()

        y_pred = y_pred.cpu().numpy()

        preds = np.concatenate([preds, y_pred])

    return preds
device = torch.device("cuda:0")
# calculate validation accuracy

val_preds = predict(best_model, val_loader, n_class=n_class, device=device)

val_acc = accuracy_score(val_df.sirna, np.argmax(val_preds, axis=1))

print(f'val acc: {val_acc}')
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

test_dataset = RCICDataset(test_df, DATA_DIR, mode='test')

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, 

                                          shuffle=False, pin_memory=True)
# predict test data

test_preds = predict(best_model, test_loader, n_class=n_class, device=device)
# save submission csv

submission_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

submission_df.sirna = np.argmax(test_preds, axis=1)

submission_df.to_csv('submission.csv', index=False)
submission_df
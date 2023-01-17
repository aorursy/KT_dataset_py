# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/proteinproject"))

print(os.listdir("../input/human-protein-atlas-image-classification"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/human-protein-resnet50-training-pytorch/saved/Protein_Resnet50/0419_084007"))
# import module we'll need to import our custom module

from shutil import copyfile

import distutils

from distutils import dir_util



distutils.dir_util.copy_tree("../input/proteinproject/project", "../working")



print(os.listdir("../working"))

%%writefile ../working/config.json

{

    "name": "Protein_Resnet50",

    "n_gpu": 1,

    

    "arch": {

        "type": "Resnet50Model",

        "args": {}

    },

    "data_loader": {

        "type": "ProteinDataLoader",

        "args":{

            "data_dir": "../input/human-protein-atlas-image-classification/train",

            "csv_path": "../input/human-protein-atlas-image-classification/train.csv",

            "img_size": 512,

            "batch_size": 25,

            "shuffle": false,

            "validation_split": 0.15,

            "num_workers": 0,

            "num_classes": 28

        }

    },

    "optimizer": {

        "type": "Adam",

        "args":{

            "lr": 0.0001,

            "amsgrad": true

        }

    },

    "loss": "focal_loss",

    "metrics": [],

    "lr_scheduler": {

        "type": "StepLR",

        "args": {

            "step_size": 2,

            "gamma": 0.1

        }

    },

    "trainer": {

        "epochs": 9,

        "save_dir": "../working/saved/",

        "save_period": 3,

        "verbosity": 2,

        

        "monitor": "min val_loss",

        "early_stop": 5,

        

        "tensorboardX": false,

        "log_dir": "../working/saved/runs"

    }

}
f = open("../working/config.json", "r")

print(f.read())

f.close()
from torchvision import datasets, transforms

from base import BaseDataLoader

from PIL import Image

import numpy as np

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms as T

from imgaug import augmenters as iaa

import pandas as pd

import pathlib

from data_loader.data_loaders import ProteinDataset





class ProteinDataLoader2(BaseDataLoader):

    def __init__(self, data_dir, csv_path, batch_size, shuffle, validation_split, num_workers, num_classes, img_size, training=True):

        self.images_df = pd.read_csv(csv_path)

        self.num_classes = num_classes

        self.dataset = ProteinDataset(self.images_df, data_dir, num_classes, img_size, not training, training)

        self.n_samples = len(self.dataset)

        super(ProteinDataLoader2, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



    def _split_sampler(self, split):

        if split == 0.0:

            return None, None



        # Dumb stratification.

        validation_split = []

        for idx, (value, count) in enumerate(self.images_df['Target'].value_counts().to_dict().items()):

            if count > 1:

                for _ in range(max(round(split * count), 1)):

                    validation_split.append(value)



        # Oversampling.

        multi = [0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4]

        validation_split_idx = []

        train_split_idx = []

        for idx, value in enumerate(self.images_df['Target']):

            try:

                validation_split.remove(value)

                validation_split_idx.append(idx)

            except:

                for _ in range(max(sum([multi[int(v)] for v in value.split(' ')]), 1)):

                    train_split_idx.append(idx)



        valid_idx = np.array(validation_split_idx)

        train_idx = np.array(train_split_idx)



        train_sampler = SubsetRandomSampler(train_idx)

        valid_sampler = SubsetRandomSampler(valid_idx)



        # turn off shuffle option which is mutually exclusive with sampler

        self.shuffle = False

        self.n_samples = len(train_idx)



        return train_sampler, valid_sampler

import torch

import torch.nn as nn

from base import BaseModel

import torchvision.models as models





class Resnet50Model(BaseModel):

    def __init__(self, num_classes=28):

        super(Resnet50Model, self).__init__()

        self.resnet = models.resnet50(pretrained=False)

        w = self.resnet.conv1.weight

        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.resnet.conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))

        self.resnet.fc = nn.Sequential(

            nn.BatchNorm1d(512 * 4),

            nn.Dropout(0.5),

            nn.Linear(512 * 4, num_classes),

        )



    def forward(self, x):

        return self.resnet(x)
import os

import json

import argparse

import torch

import data_loader.data_loaders as module_data

import model.loss as module_loss

import model.metric as module_metric

import model.resnet as module_arch

from trainer import Trainer

from utils import Logger





def get_instance(module, name, config, *args):

    return getattr(module, config[name]['type'])(*args, **config[name]['args'])





def train(config, resume):

    train_logger = Logger()



    # setup data_loader instances

    data_loader = ProteinDataLoader2(**config['data_loader']['args'])

    valid_data_loader = data_loader.split_validation()



    # build model architecture

    model = Resnet50Model()

    # load state dict

    checkpoint = torch.load("../input/human-protein-resnet50-training-pytorch/saved/Protein_Resnet50/0419_084007/model_best.pth")

    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:

        model = torch.nn.DataParallel(trained_model)

    model.load_state_dict(state_dict)

    print(model)



    # get function handles of loss and metrics

    loss = getattr(module_loss, config['loss'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]



    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)



    trainer = Trainer(model, loss, metrics, optimizer,

                      resume=resume,

                      config=config,

                      data_loader=data_loader,

                      valid_data_loader=valid_data_loader,

                      lr_scheduler=lr_scheduler,

                      train_logger=train_logger)



    trainer.train()

    

    return model





# Run!

config = json.load(open("../working/config.json"))

path = os.path.join(config['trainer']['save_dir'], config['name'])



trained_model = train(config, None)
%%writefile ../working/config-test.json

{

    "name": "Protein_Resnet50",

    "n_gpu": 1,

    

    "arch": {

        "type": "Resnet50Model",

        "args": {}

    },

    "data_loader": {

        "type": "ProteinDataLoader",

        "args":{

            "data_dir": "../input/human-protein-atlas-image-classification/test",

            "csv_path": "../input/human-protein-atlas-image-classification/sample_submission.csv",

            "img_size": 512,

            "batch_size": 1,

            "shuffle": false,

            "validation_split": 0.1,

            "num_workers": 0,

            "num_classes": 28

        }

    },

    "optimizer": {

        "type": "Adam",

        "args":{

            "lr": 0.0001,

            "amsgrad": true

        }

    },

    "loss": "focal_loss",

    "metrics": [],

    "lr_scheduler": {

        "type": "StepLR",

        "args": {

            "step_size": 2,

            "gamma": 0.1

        }

    },

    "trainer": {

        "epochs": 9,

        "save_dir": "../working/saved/",

        "save_period": 4,

        "verbosity": 2,

        

        "monitor": "min val_loss",

        "early_stop": 5,

        

        "tensorboardX": false,

        "log_dir": "../working/saved/runs"

    },

    "input_csv": "../input/human-protein-atlas-image-classification/sample_submission.csv"

}
import os

import json

import argparse

import torch

from tqdm import tqdm

import data_loader.data_loaders as module_data

import model.resnet as module_arch

import numpy as np

import pandas as pd



def test(config, resume_model):

    # setup data_loader instances

    data_loader = getattr(module_data, config['data_loader']['type'])(

        config['data_loader']['args']['data_dir'],

        config['data_loader']['args']['csv_path'],

        img_size=config['data_loader']['args']['img_size'],

        num_classes=config['data_loader']['args']['num_classes'],

        batch_size=1,

        shuffle=False,

        validation_split=0.0,

        training=False,

        num_workers=0

    )



    # build model architecture

    model = resume_model

    model.summary()



    # prepare model for testing

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    model.eval()



    sample_submission = pd.read_csv(config['input_csv'])



    os.makedirs("./submit", exist_ok=True)



    thresholds = [0.2, 0.4, 0.5]

    for threshold in thresholds:

        filenames, labels, submissions = [], [], []

        with torch.no_grad():

            for i, (data, target) in enumerate(tqdm(data_loader)):

                data = data.to(device)

                output = model(data)

                label = output.sigmoid().cpu().data.numpy()



                filenames.append(target)

                labels.append(label > threshold)



        for row in np.concatenate(labels):

            subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))

            submissions.append(subrow)

        sample_submission['Predicted'] = submissions

        sample_submission.to_csv("./submit/submission-{0:.2f}.csv".format(threshold), index=None)

        

# Run!

config = json.load(open("../working/config-test.json"))

path = os.path.join(config['trainer']['save_dir'], config['name'])



# No testing for 128x128

test(config, trained_model)

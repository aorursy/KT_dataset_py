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
print(os.listdir("../input/human-protein-resnet34-training-pytorch/saved/Protein_Resnet34/0418_055339"))
# import module we'll need to import our custom module

from shutil import copyfile

import distutils

from distutils import dir_util



distutils.dir_util.copy_tree("../input/proteinproject/project", "../working")



print(os.listdir("../working"))

%%writefile ../working/config.json

{

    "name": "Protein_Resnet34",

    "n_gpu": 1,

    

    "arch": {

        "type": "Resnet34Model",

        "args": {}

    },

    "data_loader": {

        "type": "ProteinDataLoader",

        "args":{

            "data_dir": "../input/human-protein-atlas-image-classification/train",

            "csv_path": "../input/human-protein-atlas-image-classification/train.csv",

            "img_size": 512,

            "batch_size": 50,

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

        "epochs": 16,

        "save_dir": "../working/saved/",

        "save_period": 4,

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
import torch

import torch.nn as nn

import torch.utils.model_zoo as model_zoo

from base import BaseModel

from model.resnet import ResNet, BasicBlock



model_urls = {

    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',

    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',

    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',

    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',

}



class ResNetRGBY2(ResNet):

    def __init__(self, block, layers, num_classes=28):

        super(ResNetRGBY2, self).__init__(block, layers)

        self.num_classes = num_classes



    def adopt(self):

        w = self.conv1.weight

        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1.weight = nn.Parameter(torch.cat((w, 0.5 * (w[:, :1, :, :] + w[:, 2:, :, :])), dim=1))

        self.fc = nn.Sequential(

            nn.BatchNorm1d(512),

            nn.Dropout(0.2),

            nn.Linear(512 * self.expansion, self.num_classes),

        )



    def forward(self, x):

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.maxpool(out)



        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)



        out = self.avgpool(out)

        out = out.view(out.size(0), -1)



        return self.fc(out)





class Resnet34Model2(BaseModel):

    def __init__(self, num_classes=28):

        super(Resnet34Model2, self).__init__()

        self.resnet = ResNetRGBY2(BasicBlock, [3, 4, 6, 3], num_classes)

        #self.resnet.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

        self.resnet.adopt()



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

    data_loader = get_instance(module_data, 'data_loader', config)

    valid_data_loader = data_loader.split_validation()



    # build model architecture

    model = Resnet34Model2()

    # load state dict

    checkpoint = torch.load("../input/human-protein-resnet34-training-pytorch/saved/Protein_Resnet34/0418_055339/checkpoint-epoch10.pth")

    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:

        model = torch.nn.DataParallel(trained_model)

    model.load_state_dict(state_dict)

    print(model)



    # get function handles of loss and metrics

    loss = getattr(module_loss, config['loss'])

    metrics = [getattr(module_metric, met) for met in config['metrics']]



    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler

    trainable_params = model.parameters()

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

    "name": "Protein_Resnet34",

    "n_gpu": 1,

    

    "arch": {

        "type": "Resnet34Model",

        "args": {}

    },

    "data_loader": {

        "type": "ProteinDataLoader",

        "args":{

            "data_dir": "../input/human-protein-atlas-image-classification/test",

            "csv_path": "../input/human-protein-atlas-image-classification/sample_submission.csv",

            "img_size": 512,

            "batch_size": 50,

            "shuffle": false,

            "validation_split": 0.1,

            "num_workers": 0,

            "num_classes": 28

        }

    },

    "optimizer": {

        "type": "SGD",

        "args":{

            "lr": 0.0001

        }

    },

    "loss": "focal_loss",

    "metrics": [],

    "lr_scheduler": {

        "type": "CosineAnnealingLR",

        "args": {

            "T_max": 15

        }

    },

    "trainer": {

        "epochs": 15,

        "save_dir": "../working/saved/",

        "save_period": 5,

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



    thresholds = [0.4, 0.5]

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



test(config, trained_model)

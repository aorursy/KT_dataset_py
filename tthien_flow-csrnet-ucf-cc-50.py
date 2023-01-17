# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATASET = "/kaggle/input/ucf-cc-50-with-people-density-map/ucfcrowdcountingdataset_cvpr13_with_people_density_map/UCF_CC_50/"

print(os.listdir(DATASET))
#
import os

import glob

from sklearn.model_selection import train_test_split

import json

import random

import os

from PIL import Image, ImageFilter, ImageDraw

import numpy as np

import h5py

from PIL import ImageStat

import cv2

import os

import random

import torch

import numpy as np

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms.functional as F

from torchvision import datasets, transforms



"""

create a list of file (full directory)

"""



def create_training_image_list(data_path):

    """

    create a list of absolutely path of jpg file

    :param data_path: must contain subfolder "images" with *.jpg  (example ShanghaiTech/part_A/train_data/)

    :return:

    """

    DATA_PATH = data_path

    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))

    return image_path_list





def get_train_val_list(data_path, test_size=0.1):

    DATA_PATH = data_path

    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))

    if len(image_path_list) is 0:

        image_path_list = glob.glob(os.path.join(DATA_PATH, "*.jpg"))

    train, val = train_test_split(image_path_list, test_size=test_size)



    print("train size ", len(train))

    print("val size ", len(val))

    return train, val





def load_data(img_path, train=True):

    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground-truth-h5')

    img = Image.open(img_path).convert('RGB')

    gt_file = h5py.File(gt_path, 'r')

    target = np.asarray(gt_file['density'])



    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),

                        interpolation=cv2.INTER_CUBIC) * 64



    return img, target





def load_data_ucf_cc50(img_path, train=True):

    gt_path = img_path.replace('.jpg', '.h5')

    img = Image.open(img_path).convert('RGB')

    gt_file = h5py.File(gt_path, 'r')

    target = np.asarray(gt_file['density'])



    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)),

                        interpolation=cv2.INTER_CUBIC) * 64



    return img, target



class ListDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,

                 num_workers=4, dataset_name="shanghaitech"):

        """

        if you have different image size, then batch_size must be 1

        :param root:

        :param shape:

        :param shuffle:

        :param transform:

        :param train:

        :param seen:

        :param batch_size:

        :param num_workers:

        """

        if train:

            root = root * 4

        if shuffle:

            random.shuffle(root)



        self.nSamples = len(root)

        self.lines = root

        self.transform = transform

        self.train = train

        self.shape = shape

        self.seen = seen

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.dataset_name = dataset_name

        # load data fn

        if dataset_name is "shanghaitech":

            self.load_data_fn = load_data

        elif dataset_name is "ucf_cc_50":

            self.load_data_fn = load_data_ucf_cc50



    def __len__(self):

        return self.nSamples



    def __getitem__(self, index):

        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = self.load_data_fn(img_path, self.train)

        if self.transform is not None:

            img = self.transform(img)

        return img, target





def get_dataloader(train_list, val_list, test_list, dataset_name="shanghaitech"):

    train_loader = torch.utils.data.DataLoader(

        ListDataset(train_list,

                            shuffle=True,

                            transform=transforms.Compose([

                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                            std=[0.229, 0.224, 0.225]),

                            ]),

                            train=True,

                            batch_size=1,

                            num_workers=4, dataset_name=dataset_name),

        batch_size=1, num_workers=4)



    val_loader = torch.utils.data.DataLoader(

        ListDataset(val_list,

                            shuffle=False,

                            transform=transforms.Compose([

                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                            std=[0.229, 0.224, 0.225]),

                            ]), train=False, dataset_name=dataset_name),

        batch_size=1)

    if test_list is not None:

        test_loader = torch.utils.data.DataLoader(

            ListDataset(test_list,

                        shuffle=False,

                        transform=transforms.Compose([

                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                        std=[0.229, 0.224, 0.225]),

                        ]), train=False, dataset_name=dataset_name),

            batch_size=1)

    else:

        test_loader = None



    return train_loader, val_loader, test_loader
"""

contain dummy args with config

helpfull for copy paste Kaggle

"""

import argparse





def make_args(gpu="0", task="task_one_"):

    """

    these arg does not have any required commandline arg (all with default value)

    :param train_json:

    :param test_json:

    :param pre:

    :param gpu:

    :param task:

    :return:

    """

    parser = argparse.ArgumentParser(description='PyTorch CSRNet')



    args = parser.parse_args()

    args.gpu = gpu

    args.task = task

    args.pre = None

    return args



class Meow():

    def __init__(self):

        pass





def make_meow_args(gpu="0", task="task_one_"):

    args = Meow()

    args.gpu = gpu

    args.task = task

    args.pre = None

    return args





def like_real_args_parse(data_input):

    args = Meow()

    args.input = data_input

    args.original_lr = 1e-7

    args.lr = 1e-7

    args.batch_size = 1

    args.momentum = 0.95

    args.decay = 5 * 1e-4

    args.start_epoch = 0

    args.epochs = 120

    args.steps = [-1, 1, 100, 150]

    args.scales = [1, 1, 1, 1]

    args.workers = 4

    args.print_freq = 30

    return args





def real_args_parse():

    """

    this is not dummy

    if you are going to make all-in-one notebook, ignore this

    :return:

    """

    parser = argparse.ArgumentParser(description='CrowdCounting')

    parser.add_argument("--task_id", action="store", default="dev")

    parser.add_argument('-a', action="store_true", default=False)



    parser.add_argument('--input', action="store",  type=str)

    parser.add_argument('--output', action="store", type=str)

    parser.add_argument('--model', action="store", default="csrnet")



    # args with default value

    parser.add_argument('--lr', action="store", default=1e-7, type=float)

    parser.add_argument('--momentum', action="store", default=0.95, type=float)

    parser.add_argument('--decay', action="store", default=5*1e-4, type=float)

    parser.add_argument('--epochs', action="store", default=1, type=int)



    # args.original_lr = 1e-7

    # args.lr = 1e-7

    # args.batch_size = 1

    # args.momentum = 0.95

    # args.decay = 5 * 1e-4

    # args.start_epoch = 0

    # args.epochs = 120

    # args.steps = [-1, 1, 100, 150]

    # args.scales = [1, 1, 1, 1]

    # args.workers = 4

    # args.seed = time.time()

    # args.print_freq = 30



    arg = parser.parse_args()

    return arg
from __future__ import division



import torch

import math

from ignite.exceptions import NotComputableError

from ignite.metrics.metric import Metric



class CrowdCountingMeanAbsoluteError(Metric):

    """

    Calculates the mean absolute error.



    - `update` must receive output of the form `(y_pred, y)`.

    """

    def reset(self):

        self._sum_of_absolute_errors = 0.0

        self._num_examples = 0



    def update(self, output):

        y_pred, y = output

        pred_count = torch.sum(y_pred)

        true_count = torch.sum(y)

        absolute_errors = torch.abs(pred_count - true_count)

        self._sum_of_absolute_errors += torch.sum(absolute_errors).item()

        self._num_examples += y.shape[0]



    def compute(self):

        if self._num_examples == 0:

            raise NotComputableError('MeanAbsoluteError must have at least one example before it can be computed.')

        return self._sum_of_absolute_errors / self._num_examples





class CrowdCountingMeanSquaredError(Metric):

    """

    Calculates the mean squared error.



    - `update` must receive output of the form `(y_pred, y)`.

    """

    def reset(self):

        self._sum_of_squared_errors = 0.0

        self._num_examples = 0



    def update(self, output):

        y_pred, y = output

        pred_count = torch.sum(y_pred)

        true_count = torch.sum(y)

        squared_errors = torch.pow(pred_count - true_count, 2)

        self._sum_of_squared_errors += torch.sum(squared_errors).item()

        self._num_examples += y.shape[0]



    def compute(self):

        if self._num_examples == 0:

            raise NotComputableError('MeanSquaredError must have at least one example before it can be computed.')

        return math.sqrt(self._sum_of_squared_errors / self._num_examples)





import torch.nn as nn

import torch

from torchvision import models





class CSRNet(nn.Module):

    def __init__(self, load_weights=False):

        super(CSRNet, self).__init__()

        self.seen = 0

        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]

        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)

        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:

            mod = models.vgg16(pretrained=True)

            self._initialize_weights()

            for i in range(len(list(self.frontend.state_dict().items()))):

                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]



    def forward(self, x):

        x = self.frontend(x)

        x = self.backend(x)

        x = self.output_layer(x)



        # remove channel dimension

        # (N, C_{out}, H_{out}, W_{out}) => (N, H_{out}, W_{out})

        x = torch.squeeze(x, dim=1)

        return x



    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.normal_(m.weight, std=0.01)

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)





def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):

    if dilation:

        d_rate = 2

    else:

        d_rate = 1

    layers = []

    for v in cfg:

        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)

            if batch_norm:

                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            else:

                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)




from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Loss, MeanAbsoluteError, MeanSquaredError



import torch

from torch import nn



import os





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

args = like_real_args_parse(DATASET)

print(args)

DATA_PATH = args.input





# create list

train_list, val_list = get_train_val_list(DATA_PATH, test_size=0.2)

test_list = None



# create data loader

train_loader, val_loader, test_loader = get_dataloader(train_list, val_list, test_list, dataset_name="ucf_cc_50")





# model

model = CSRNet()

model = model.to(device)



# loss function

loss_fn = nn.MSELoss(size_average=False).cuda()



optimizer = torch.optim.SGD(model.parameters(), args.lr,

                            momentum=args.momentum,

                            weight_decay=args.decay)



trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

evaluator = create_supervised_evaluator(model,

                                        metrics={

                                            'mae': CrowdCountingMeanAbsoluteError(),

                                            'mse': CrowdCountingMeanSquaredError(),

                                            'nll': Loss(loss_fn)

                                        }, device=device)

print(model)





@trainer.on(Events.ITERATION_COMPLETED)

def log_training_loss(trainer):

    print("Epoch[{}] Interation [{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.iteration, trainer.state.output))





@trainer.on(Events.EPOCH_COMPLETED)

def log_training_results(trainer):

    evaluator.run(train_loader)

    metrics = evaluator.state.metrics

    print("Validation Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"

          .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))





@trainer.on(Events.EPOCH_COMPLETED)

def log_validation_results(trainer):

    evaluator.run(val_loader)

    metrics = evaluator.state.metrics

    print("Validation Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"

          .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))





trainer.run(train_loader, max_epochs=150)
evaluator.run(val_loader)

metrics = evaluator.state.metrics

print("Validation Results - Epoch: {}  Avg mae: {:.2f} Avg mse: {:.2f} Avg loss: {:.2f}"

      .format(trainer.state.epoch, metrics['mae'], metrics['mse'], metrics['nll']))
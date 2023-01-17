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
print(os.listdir("../input/shanghaitech_with_people_density_map/ShanghaiTech"))
DATA_PATH = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_B/train_data/"

TEST_DATA_PATH = "../input/shanghaitech_with_people_density_map/ShanghaiTech/part_B/test_data/"
PRETRAINED_MODEL = None
import h5py

import torch

import shutil



def save_net(fname, net):

    with h5py.File(fname, 'w') as h5f:

        for k, v in net.state_dict().items():

            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):

    with h5py.File(fname, 'r') as h5f:

        for k, v in net.state_dict().items():        

            param = torch.from_numpy(np.asarray(h5f[k]))         

            v.copy_(param)

            

def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):

    torch.save(state, task_id+filename)

    if is_best:

        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')        
"""

contain dummy args with config

helpfull for copy paste Kaggle

"""

import argparse





def make_args(train_json= "", test_json="", pre="", gpu="0", task="task_one_"):

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
import os

import glob

from sklearn.model_selection import train_test_split

import json

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





def get_train_val_list(data_path):

    DATA_PATH = data_path

    image_path_list = glob.glob(os.path.join(DATA_PATH, "images", "*.jpg"))

    train, val = train_test_split(image_path_list, test_size=0.3, random_state=113)



    print("train size ", len(train))

    print("val size ", len(val))

    return train, val
import random

import os

from PIL import Image,ImageFilter,ImageDraw

import numpy as np

import h5py

from PIL import ImageStat

import cv2



def load_data(img_path,train = True):

    gt_path = img_path.replace('.jpg','.h5').replace('images','ground-truth-h5')

    img = Image.open(img_path).convert('RGB')

    gt_file = h5py.File(gt_path, 'r')

    target = np.asarray(gt_file['density'])



    target = cv2.resize(target,(int(target.shape[1]/8), int(target.shape[0]/8)),interpolation = cv2.INTER_CUBIC)*64

    

    return img,target
import os

import random

import torch

import numpy as np

from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms.functional as F





class ListDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):

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

            root = root *4

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

        

    def __len__(self):

        return self.nSamples



    def __getitem__(self, index):

        assert index <= len(self), 'index range error' 

        

        img_path = self.lines[index]

        

        img,target = load_data(img_path,self.train)

        

        #img = 255.0 * F.to_tensor(img)

        

        #img[0,:,:]=img[0,:,:]-92.8207477031

        #img[1,:,:]=img[1,:,:]-95.2757037428

        #img[2,:,:]=img[2,:,:]-104.877445883





        

        

        if self.transform is not None:

            img = self.transform(img)

        return img,target

import torch.nn as nn

import torch

from torchvision import models


def create_conv2d_block(in_channels, kernel_size, n_filter, dilated_rate=1):

    # padding formula  https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338

    """

    o = output

    p = padding

    k = kernel_size

    s = stride

    d = dilation

    """

#     o = [i + 2*p - k - (k-1)*(d-1)]/s + 1

    k = kernel_size

    d = dilated_rate

    padding_rate = int((k + (k-1)*(d-1))/2)

    conv2d =  nn.Conv2d(in_channels, n_filter, kernel_size, padding=padding_rate, dilation = dilated_rate)

    bn = nn.BatchNorm2d(n_filter)

    relu = nn.ReLU(inplace=True)

    return [conv2d, bn, relu]
class ScalePyramidModule(nn.Module):

    def __init__(self):

        super(ScalePyramidModule, self).__init__()

        self.a = nn.Sequential(*create_conv2d_block(512, 3, 512, 2))

        self.b = nn.Sequential(*create_conv2d_block(512, 3, 512, 4))

        self.c = nn.Sequential(*create_conv2d_block(512, 3, 512, 8))

        self.d = nn.Sequential(*create_conv2d_block(512, 3, 512, 12))

    def forward(self,x):

        xa = self.a(x)

        xb = self.b(x)

        xc = self.c(x)

        xd = self.d(x)

        return torch.cat((xa, xb, xc, xd), 1)
def make_layers_by_cfg(cfg, in_channels = 3,batch_norm=True, dilation = True):

    """

    cfg: list of tuple (number of layer, kernel, n_filter, dilated) or 'M'

    """

    if dilation:

        d_rate = 2

    else:

        d_rate = 1

    layers = []

    for v in cfg:

        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:

            # number of layer, kernel, n_filter, dilated

            for t in range(v[0]):

              layers += create_conv2d_block(in_channels, v[1], v[2], v[3])

              in_channels = v[2]

    return nn.Sequential(*layers)                
class SPN(nn.Module):

    def __init__(self):

        super(SPN, self).__init__()

        self.frontend_config = [(2,3,64,1), 'M', (2,3,128,1), 'M', (2,3,256,1), 'M', (3,3,512,1)] 

        self.backend_config = [(1,3,256,1), (1,3,512,1)]

        self.frontend = make_layers_by_cfg(self.frontend_config)

        self.spm = ScalePyramidModule()

        self.backend = make_layers_by_cfg(self.backend_config, 512*4)

        self.output_layer = nn.Sequential(*create_conv2d_block(512, 1, 1, 1))

        self.seen = 0

    def forward(self,x):

        x1 = self.frontend(x)

        x2 = self.spm(x1)

        x3 = self.backend(x2)

        output = self.output_layer(x3)

        return output
spn = SPN()

print(spn)
x = torch.rand(1, 3, 224, 224)

out = spn(x)

print(out.shape)
import sys

import os



import warnings



# import from library

import torch

import torch.nn as nn

from torch.autograd import Variable

from torchvision import datasets, transforms

import numpy as np

import argparse

import json

import cv2

import time



"""

A dublicate of train.py 

However it does not need commandline arg

"""







def main():

    global args, best_prec1

    args = make_meow_args()





    best_prec1 = 1e6



    args.original_lr = 1e-7

    args.lr = 1e-7

    args.batch_size = 1

    args.momentum = 0.95

    args.decay = 5 * 1e-4

    args.start_epoch = 0

    args.epochs = 10

    args.steps = [-1, 1, 100, 150]

    args.scales = [1, 1, 1, 1]

    args.workers = 4

    args.seed = time.time()

    args.print_freq = 30

    args.pre = PRETRAINED_MODEL



    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.cuda.manual_seed(args.seed)



    train_list, val_list = get_train_val_list(DATA_PATH)



    model = SPN()



    model = model.cuda()



    criterion = nn.MSELoss(size_average=False).cuda()



    optimizer = torch.optim.SGD(model.parameters(), args.lr,

                                momentum=args.momentum,

                                weight_decay=args.decay)



    if args.pre:

        if os.path.isfile(args.pre):

            print("=> loading checkpoint '{}'".format(args.pre))

            checkpoint = torch.load(args.pre)

            args.start_epoch = checkpoint['epoch']

            best_prec1 = checkpoint['best_prec1']

            model.load_state_dict(checkpoint['state_dict'])

            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"

                  .format(args.pre, checkpoint['epoch']))

        else:

            print("=> no checkpoint found at '{}'".format(args.pre))



    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)



        train(train_list, model, criterion, optimizer, epoch)

        prec1 = validate(val_list, model, criterion)



        is_best = prec1 < best_prec1

        best_prec1 = min(prec1, best_prec1)

        print(' * best MAE {mae:.3f} '

              .format(mae=best_prec1))

        save_checkpoint({

            'epoch': epoch + 1,

            'arch': args.pre,

            'state_dict': model.state_dict(),

            'best_prec1': best_prec1,

            'optimizer': optimizer.state_dict(),

        }, is_best, args.task)





def train(train_list, model, criterion, optimizer, epoch):

    losses = AverageMeter()

    batch_time = AverageMeter()

    data_time = AverageMeter()



    train_loader = torch.utils.data.DataLoader(

        ListDataset(train_list,

                            shuffle=True,

                            transform=transforms.Compose([

                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                            std=[0.229, 0.224, 0.225]),

                            ]),

                            train=True,

                            seen=model.seen,

                            batch_size=args.batch_size,

                            num_workers=args.workers),

        batch_size=args.batch_size)

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))



    model.train()

    end = time.time()



    for i, (img, target) in enumerate(train_loader):

        data_time.update(time.time() - end)



        img = img.cuda()

        img = Variable(img)

        output = model(img)



        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()

        target = Variable(target)



        loss = criterion(output, target)



        losses.update(loss.item(), img.size(0))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        batch_time.update(time.time() - end)

        end = time.time()



        if i % args.print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'

                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'

                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'

                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'

                .format(

                epoch, i, len(train_loader), batch_time=batch_time,

                data_time=data_time, loss=losses))





def validate(val_list, model, criterion):

    print('begin test')

    test_loader = torch.utils.data.DataLoader(

        ListDataset(val_list,

                            shuffle=False,

                            transform=transforms.Compose([

                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],

                                                                            std=[0.229, 0.224, 0.225]),

                            ]), train=False),

        batch_size=args.batch_size)



    model.eval()



    mae = 0



    for i, (img, target) in enumerate(test_loader):

        img = img.cuda()

        img = Variable(img)

        output = model(img)



        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())



    mae = mae / len(test_loader)

    print(' * MAE {mae:.3f} '

          .format(mae=mae))



    return mae





def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""



    args.lr = args.original_lr



    for i in range(len(args.steps)):



        scale = args.scales[i] if i < len(args.scales) else 1



        if epoch >= args.steps[i]:

            args.lr = args.lr * scale

            if epoch == args.steps[i]:

                break

        else:

            break

    for param_group in optimizer.param_groups:

        param_group['lr'] = args.lr





class AverageMeter(object):

    """Computes and stores the average and current value"""



    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





if __name__ == '__main__':

    main()
test_image_list = create_training_image_list(TEST_DATA_PATH)
best_checkpoint = torch.load("task_one_checkpoint.pth.tar")

model = SPN()

optimizer = torch.optim.SGD(model.parameters(), args.lr,

                            momentum=args.momentum,

                            weight_decay=args.decay)

criterion = nn.MSELoss(size_average=False).cuda()

model.load_state_dict(best_checkpoint['state_dict'])

optimizer.load_state_dict(best_checkpoint['optimizer'])

model = model.cuda()
test_result = validate(test_image_list, model, criterion)
best_checkpoint = torch.load("task_one_model_best.pth.tar")

model = SPN()

optimizer = torch.optim.SGD(model.parameters(), args.lr,

                            momentum=args.momentum,

                            weight_decay=args.decay)

criterion = nn.MSELoss(size_average=False).cuda()

model.load_state_dict(best_checkpoint['state_dict'])

optimizer.load_state_dict(best_checkpoint['optimizer'])

model = model.cuda()
test_result = validate(test_image_list, model, criterion)
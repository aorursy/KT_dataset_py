!pip install -q efficientnet_pytorch
import os

import time

import random

import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps, ImageDraw

from torchvision import transforms, models

from efficientnet_pytorch import EfficientNet

import torch

from torch import optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch import nn

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import WeightedRandomSampler

from torch import cuda
#--------------------

# utils

#--------------------

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    



class AverageMeter():

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
#--------------------

# augmentations

#--------------------

class UniformAugment():

    def __init__(self, NumOps=2, fillcolor=(128, 128, 128)):

        self.NumOps = NumOps

        self.augs = {

            'shearX': [-0.3, 0.3],

            'shearY': [-0.3, 0.3],

            'translateX': [-0.45, 0.45],

            'translateY': [-0.45, 0.45],

            'rotate': [-30, 30],

            'autocontrast': [0, 0],

            'invert': [0, 0],

            'equalize': [0, 0],

            'solarize': [0, 256],

            'posterize': [4, 8],

            'contrast': [0.1, 1.9],

            'color': [0.1, 1.9],

            'brightness': [0.1, 1.9],

            'sharpness': [0.1, 1.9],

            'cutout': [0, 0.2] 

        }



        def rotate_with_fill(img, magnitude):

            rot = img.convert('RGBA').rotate(magnitude)

            return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)



        def cutout(img, magnitude, fillcolor):

            img = img.copy()

            w, h = img.size

            v = w * magnitude

            x0 = np.random.uniform(w)

            y0 = np.random.uniform(h)

            x0 = int(max(0, x0 - v / 2.))

            y0 = int(max(0, y0 - v / 2.))

            x1 = min(w, x0 + v)

            y1 = min(h, y0 + v)

            xy = (x0, y0, x1, y1)

            ImageDraw.Draw(img).rectangle(xy, fillcolor)

            return img



        self.func = {

            'shearX': lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, magnitude, 0, 0, 1, 0), Image.BICUBIC, fillcolor=fillcolor),

            'shearY': lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, magnitude, 1, 0), Image.BICUBIC, fillcolor=fillcolor),

            'translateX': lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, magnitude, 0, 1, 0), fillcolor=fillcolor),

            'translateY': lambda img, magnitude: img.transform(

                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude), fillcolor=fillcolor),

            'rotate': lambda img, magnitude: rotate_with_fill(img, magnitude),

            'color': lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),

            'posterize': lambda img, magnitude: ImageOps.posterize(img, int(magnitude)),

            'solarize': lambda img, magnitude: ImageOps.solarize(img, int(magnitude)),

            'contrast': lambda img, magnitude: ImageEnhance.Contrast(img).enhance(magnitude),

            'sharpness': lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(magnitude),

            'brightness': lambda img, magnitude: ImageEnhance.Brightness(img).enhance(magnitude),

            'autocontrast': lambda img, magnitude: ImageOps.autocontrast(img),

            'equalize': lambda img, magnitude: ImageOps.equalize(img),

            'invert': lambda img, magnitude: ImageOps.invert(img),

            'cutout': lambda img, magnitude: cutout(img, magnitude, fillcolor=fillcolor)

        }



    def __call__(self, img):

        operations = random.sample(list(self.augs.items()), self.NumOps)

        for operation in operations:

            aug, range = operation

            magnitude = random.uniform(range[0], range[1])

            probability = random.random()

            if random.random() < probability:

                img = self.func[aug](img, magnitude)

        return img

  

  

class ImageTransform():

    def __init__(self, resize, uniform_augment, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), train=True):

        self.data_transform = {

            'train': transforms.Compose([

                transforms.RandomResizedCrop(size=resize, scale=(0.7, 1.0)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomVerticalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ]),

            'valid': transforms.Compose([

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ]),

            'test': transforms.Compose([

                transforms.RandomResizedCrop(size=resize, scale=(0.7, 1.0)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomVerticalFlip(),

                transforms.ToTensor(),

                transforms.Normalize(mean, std)

            ])

        }

        if uniform_augment:

            self.data_transform['train'].transforms.insert(0, UniformAugment())

            self.data_transform['test'].transforms.insert(0, UniformAugment())

            

    def __call__(self, img, phase):

        return self.data_transform[phase](img=img)
#--------------------

# dataset

#--------------------

class MelanomaDataset(Dataset):

    def __init__(self, base_dir, info, transform=None, phase='train'):

        self.base_dir = base_dir

        self.info = info

        self.transform = transform

        self.phase = phase



    def __len__(self):

        return len(self.info)



    def __getitem__(self, index):

        p = Path(self.base_dir, self.info.loc[index, 'image_name'] + '.jpg')

        img = Image.open(p)

        img_transformed = self.transform(img, self.phase)

        return {

            'inputs': img_transformed,

            'labels': torch.tensor(self.info.loc[index, 'target'], dtype=torch.int64)

        }
#--------------------

# models

#--------------------

def load_model(model, out_features):

    net = None

    if model.startswith('efficientnet'):

        net = EfficientNet.from_pretrained(model, num_classes=out_features)

        for name, param in net.named_parameters():

            if '_fc' in name:

                param.requires_grad = True

            else:

                param.requires_grad = False                

                

    elif model == 'vgg19':

        net = models.vgg19(pretrained=True)

        net.classifier[6] = nn.Linear(in_features=net.classifier[6].weight.size(1), out_features=out_features)

        for name, param in net.named_parameters():

            if name in ['classifier.6.weight', 'classifier.6.bias']:

                param.requires_grad = True

            else:

                param.requires_grad = False



    elif model == 'resnext':

        net = models.resnext101_32x8d(pretrained=True)

        net.fc = nn.Linear(in_features=net.fc.weight.size(1), out_features=out_features)

        for name, param in net.named_parameters():

            if 'fc' in name:

                param.requires_grad = True

            else:

                param.requires_grad = False

                

    elif model == 'densenet':

        net = models.densenet161(pretrained=True)

        net.classifier = nn.Linear(in_features=net.classifier.weight.size(1), out_features=out_features)

        for name, param in net.named_parameters():

            if 'classifier' in name:

                param.requires_grad = True

            else:

                param.requires_grad = False



    return net
#--------------------

# train and evaluate

#--------------------

def eval_model(model, dataset, batch_size, criterion, num_workers, device=None):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    losses = AverageMeter()

    preds = []

    targets = []

    with torch.no_grad():

        for data in dataloader:

            inputs = data['inputs'].to(device)

            labels = data['labels'].to(device)



            # forward

            outputs = model(inputs)



            # loss

            loss = criterion(outputs, labels)

            losses.update(loss.item(), inputs.size(0))



            # prediction

            preds.append(outputs[:, 1])

            targets.append(labels)

            

        # auc

        preds = torch.cat(preds, dim=-1).cpu().numpy()

        targets = torch.cat(targets, dim=-1).cpu().numpy()

        auc = roc_auc_score(targets, preds)

      

    return losses.avg, auc

  



def train_model(

    model_id, 

    dataset_train, 

    dataset_valid, 

    batch_size, 

    model, 

    criterion, 

    optimizer, 

    scheduler,

    num_epochs, 

    freezed_epochs,

    base_dir, 

    num_workers, 

    sampler, 

    device, 

    early_stopping

    ):



    model.to(device)

    

    # create a dataloader

    if sampler != None:

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    else:

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)



    # train

    for epoch in range(1, num_epochs + 1):

        losses = AverageMeter()

        s_time = time.time()

        

        if epoch == freezed_epochs + 1:

            # unfreeze upstream layers

            model.load_state_dict(torch.load(f'./state_dict_{model_id}.pt', map_location=device))

            for param in model.parameters():

                param.requires_grad = True

            for g in optimizer.param_groups:

                g['lr'] = 1e-4



        # set the train mode

        model.train()

        

        for data in dataloader_train:

            # zero the parameter gradients

            optimizer.zero_grad()

                        

            # forward + backward + optimize

            inputs = data['inputs'].to(device)

            labels = data['labels'].to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            losses.update(loss.item(), inputs.size(0))

        

        # set the eval mode

        model.eval()



        # calculate validation loss and auc

        loss_valid, auc_valid = eval_model(model, dataset_valid, batch_size, criterion, num_workers, device)



        # save the checkpoint

        if epoch == 1 or auc_valid > max_auc:

            saved = True

            max_auc = auc_valid

            torch.save(model.state_dict(), f'./state_dict_{model_id}.pt')

            counter = 0

        else:

            saved = False

            counter += 1



        # print statistics

        e_time = time.time()

        print(f'epoch: {epoch}, loss_train: {losses.avg:.4f}, loss_valid: {loss_valid:.4f}, auc_valid: {auc_valid:.4f}, saved: {saved}, {(e_time - s_time):.4f}sec') 



        # no operation in freezed epochs

        if epoch < freezed_epochs + 1:

            counter = 0

        else:

            # step the scheduler            

            scheduler.step(auc_valid)

    

            # early stopping

            if early_stopping != None:

                if counter == early_stopping:

                    break

        

        

def predict(dataset, batch_size, model, device=None):

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)

    model.eval()

    preds = []

    with torch.no_grad():

        for data in dataloader:

            inputs = data['inputs'].to(device)

            outputs = model(inputs)

            preds.append(outputs)

        preds = torch.cat(preds)



    return preds
# set parameters

seed_everything(123)

torch.backends.cudnn.benchmark = True



config = {

    'INPUT_DIR'      : '../input/jpeg-melanoma-256x256',

    'MODEL'          : 'efficientnet-b0',

    'SIZE'           : 256,

    'BATCH_SIZE'     : 128,

    'NUM_FOLDS'      : 5,

    'NUM_EPOCHS'     : 20,

    'FREEZED_EPOCHS' : 5,

    'LEARNING_RATE'  : 1e-3,

    'EARLY_STOPPING' : 3,

    'UNIFORM_AUGMENT': True,

    'TTA'            : 5,

    'NUM_WORKERS'    : 4,

    'DEVICE'         : 'cuda'

}
# load data

train = pd.read_csv(Path(config['INPUT_DIR'], 'train.csv'))

test = pd.read_csv(Path(config['INPUT_DIR'], 'test.csv'))

sub = pd.read_csv(Path(config['INPUT_DIR'], 'sample_submission.csv'))



# define transformer

transform = ImageTransform(config['SIZE'], config['UNIFORM_AUGMENT'])
# cross validation

prediction = pd.DataFrame()

skf = StratifiedKFold(n_splits=config['NUM_FOLDS'], shuffle=True)

for i, (tr_idx, va_idx) in enumerate(skf.split(train, train['target'])):

    tr = train.loc[tr_idx, :]

    va = train.loc[va_idx, :]

    tr.reset_index(drop=True, inplace=True)

    va.reset_index(drop=True, inplace=True)



    # create datasets

    dataset_train = MelanomaDataset(Path(config['INPUT_DIR'], 'train'), tr, transform=transform, phase='train')

    dataset_valid = MelanomaDataset(Path(config['INPUT_DIR'], 'train'), va, transform=transform, phase='valid')

    

    # load a pretrained model

    net = load_model(config['MODEL'], 2)



    # define a loss function

    criterion = nn.CrossEntropyLoss()



    # define an optimizer

    optimizer = optim.Adam(net.parameters(), lr=config['LEARNING_RATE'])



    # define a scheduler

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2, factor=0.2)



    # create a sampler

    class_sample_count = np.array([len(np.where(tr['target'] == t)[0]) for t in np.unique(tr['target'])])

    weight = 1. / class_sample_count

    samples_weight = np.array([weight[t] for t in tr['target']])

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))



    # train the network

    print(f"---- fold: {i + 1} ------------")

    train_model(

        f"{config['MODEL']}_{i + 1}",

        dataset_train,

        dataset_valid,

        config['BATCH_SIZE'],

        net,

        criterion,

        optimizer,

        scheduler,

        config['NUM_EPOCHS'],

        config['FREEZED_EPOCHS'],

        config['INPUT_DIR'],

        config['NUM_WORKERS'],

        sampler,

        config['DEVICE'],

        config['EARLY_STOPPING']

    )

    

    # predict on test dataset

    test['target'] = 0

    dataset_test = MelanomaDataset(Path(config['INPUT_DIR'], 'test'), test, transform=transform, phase='test')

    for _ in range(config['TTA']):                                        

        pred_test = predict(dataset_test, config['BATCH_SIZE'], net, device=config['DEVICE'])

        pred_test = pd.DataFrame(torch.softmax(pred_test, 1)[:, 1].cpu().numpy())

        prediction = pd.concat([prediction, pred_test], axis=1)

    

# output

sub['target'] = prediction.mean(axis=1)

sub.to_csv('./submission.csv', index=False)
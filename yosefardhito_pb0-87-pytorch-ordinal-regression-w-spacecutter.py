import random

import sys

import os

import time

import math

import gc

from functools import partial



import cv2



import numpy as np

import scipy as sp

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image



from tqdm._tqdm_notebook import tqdm_notebook as tqdm

# tqdm.pandas()

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix



import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



spacecutter_package_path = '../input/spacecutter/spacecutter-master/spacecutter-master/'

sys.path.append(spacecutter_package_path)

from spacecutter.models import OrdinalLogisticModel

from spacecutter.losses import CumulativeLinkLoss



enet_package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'

sys.path.append(enet_package_path)

from efficientnet_pytorch import EfficientNet



device = 'cuda' if torch.cuda.is_available() else 'cpu'

DEBUG = False    # always change this to 'False' before comitting,

                # you can change to 'True' during editing to use cache

                # and make subsequent training faster



def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

seed_everything(1336)
class RetinaDataset(Dataset):

    CACHE_DIR = 'cache'

    

    def __init__(self, dataframe, img_size, img_scale, train_transform, use_base_transform, use_cache=False):

        if use_cache and not os.path.exists(self.CACHE_DIR): os.mkdir(self.CACHE_DIR)

        self.use_cache = use_cache

        self.df = dataframe

        self.train_transform = train_transform

        self.img_size = img_size

        self.img_scale = img_scale

        self.epoch = 0

        self.use_base_transform = use_base_transform

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        label = self.df.diagnosis.values[idx]

        label = np.expand_dims(label, -1)

        img_path = self.df.img_path.values[idx]

        id_code = self.df.index.values[idx]

        if self.use_cache:

            filename = id_code + ".png"

            cache_path = os.path.join(self.CACHE_DIR, id_code+".png")

            cached = os.path.exists(cache_path)

            try:

                imgpil = Image.open(cache_path)

                imgpil = self.train_transform(imgpil)

            except (OSError, IOError) as e:

                imgpil = self.load_base_transform(img_path)

                imgpil.save(cache_path,"PNG")

                imgpil = self.train_transform(imgpil)

        else:

            imgpil = self.load_base_transform(img_path)

            imgpil = self.train_transform(imgpil)

        imgt = transforms.ToTensor()(imgpil)

        imgt = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(imgt)

        return imgt, label

        

    def crop_image_from_gray(self, img, tol=7):

        if img.ndim==2:

            mask = img>tol

            return img[np.ix_(mask.any(1),mask.any(0))]

        elif img.ndim==3:

            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            mask = gray_img>tol



            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

            if (check_shape == 0): # image is too dark so that we crop out everything,

                return img # return original image

            else:

                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

                img = np.stack([img1,img2,img3],axis=-1)

            return img



    def load_base_transform(self, img_path):

        if self.use_base_transform is None or len(self.use_base_transform) == 0:

            imgpil = Image.open(img_path)

            w, h = imgpil.size

            base_size = int(self.img_size * self.img_scale)

            w_new = base_size if w <= h else int(w * base_size / h)

            h_new = base_size if h <= w else int(h * base_size / w)

            return imgpil.resize((w_new, h_new), Image.ANTIALIAS)

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if 'crop' in self.use_base_transform:

            img = self.crop_image_from_gray(img)

        # resize

        w, h, _ = img.shape

        base_size = int(self.img_size * self.img_scale)

        w_new = base_size if w <= h else int(w * base_size / h)

        h_new = base_size if h <= w else int(h * base_size / w)

        img = cv2.resize(img, (h_new, w_new))

        # ben's preprocessing

        if 'weighted' in self.use_base_transform:

            img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0) , 10), -4 ,128)

        return transforms.ToPILImage()(img)

    

    def show_sample_imgs(self, n, get_original, use_train_transform, per_row=2):

        samples = self.df.sample(n=n)

        img_names = samples.index.values

        rows = (n + per_row - 1)//per_row

        cols = min(per_row, n)

        fig, axes = plt.subplots(rows, cols, figsize=(15,15))

        for ax in axes.flatten(): 

            ax.axis('off')

        for i,(img_name, ax) in enumerate(zip(img_names, axes.flatten())): 

            img_path = self.df.loc[img_name].img_path

            if get_original:

                imgpil = Image.open(img_path)

            else:

                imgpil = self.load_base_transform(img_path)

            if use_train_transform:

                imgpil = self.train_transform(imgpil)

            ax.imshow(imgpil)

            ax.set_title(self.df.loc[img_name].diagnosis)
class NNLogger(object):

    

    def __init__(self):

        # mini-batch-oriented

        self.y_true = {'train': [], 'val': []}

        self.y_pred = {'train': [], 'val': []}

        self.y_true = {'train': [], 'val': []}

        self.y_pred = {'train': [], 'val': []}

        self.loss = {'train': [], 'val': []}

        self.elapsed_time = []

        self.lr_history = []

        # epoch-oriented

        self.current_epoch = 1

    

    def step(self):

        self.current_epoch += 1

        

    def record_stat(self, lr, elapsed_time):

        self.lr_history.append(lr)

        self.elapsed_time.append(elapsed_time)

        

    def add(self, mode, y_true, y_pred, loss):

        if len(self.y_true[mode]) < self.current_epoch:

            self.y_true[mode].append([*y_true])

            self.y_pred[mode].append([*y_pred])

            self.loss[mode].append([loss])

        else:

            self.y_true[mode][self.current_epoch-1] = [*self.y_true[mode][self.current_epoch-1], *y_true]

            self.y_pred[mode][self.current_epoch-1] = [*self.y_pred[mode][self.current_epoch-1], *y_pred]

            self.loss[mode][self.current_epoch-1].append(loss)

        

    def get_last_kappa_score(self):

        return (cohen_kappa_score(self.y_true['train'][-1], self.y_pred['train'][-1], weights='quadratic'),

                cohen_kappa_score(self.y_true['val'][-1], self.y_pred['val'][-1], weights='quadratic'))

    

    def get_last_mae(self):

        return (mean_absolute_error(self.y_true['train'][-1], self.y_pred['train'][-1]),

                mean_absolute_error(self.y_true['val'][-1], self.y_pred['val'][-1]))

    

    def get_last_cm(self):

        return confusion_matrix(self.y_true['val'][-1], self.y_pred['val'][-1])
class BlindnessDetectionTrainer(object):

    # training sets

    _train_2015_csv = '../input/resized-2015-2019-blindness-detection-images/labels/trainLabels15.csv'

    _train_2015_img_path = '../input/resized-2015-2019-blindness-detection-images/resized train 15/'

    _train_2015_ext = '.jpg'

    _train_2019_csv = '../input/resized-2015-2019-blindness-detection-images/labels/trainLabels19.csv'

    _train_2019_img_path = '../input/resized-2015-2019-blindness-detection-images/resized train 19/'

    _train_2019_ext = '.jpg'



    def __init__(self, load_pretrained=True, freeze_pretrained=False, use_cache=False, test_size=0.1):

        self.use_cache = use_cache

        # load training set 2015

        self.train_2015_csv = pd.read_csv(self._train_2015_csv, names=['id_code', 'diagnosis'], skiprows=1)

        self.train_2015_csv['img_path'] = self.train_2015_csv['id_code'].apply(

            lambda f: self._train_2015_img_path + f + self._train_2015_ext if os.path.isfile(

                self._train_2015_img_path + f + self._train_2015_ext) else None)

        self.train_2015_csv = self.train_2015_csv.set_index('id_code')

        self.train_2015_df = self.train_2015_csv.dropna()

        self.train_2015_df['type'] = 'old' # indicate data from old competition (2015)

        # downsampling class 0

        class_dist_2015 = [

            7000,

            len(self.train_2015_df[self.train_2015_df['diagnosis'] == 1]),

            len(self.train_2015_df[self.train_2015_df['diagnosis'] == 2]),

            len(self.train_2015_df[self.train_2015_df['diagnosis'] == 3]),

            len(self.train_2015_df[self.train_2015_df['diagnosis'] == 4]),

        ]

        self.train_2015_df = self.resample(self.train_2015_df, class_dist_2015)

        # load training set 2019

        self.train_2019_csv = pd.read_csv(self._train_2019_csv, names=['id_code', 'diagnosis'], skiprows=1)

        self.train_2019_csv['img_path'] = self.train_2019_csv['id_code'].apply(

            lambda f: self._train_2019_img_path + f + self._train_2019_ext if os.path.isfile(

                self._train_2019_img_path + f + self._train_2019_ext) else None)

        self.train_2019_csv = self.train_2019_csv.set_index('id_code')

        self.train_2019_df = self.train_2019_csv.dropna()

        self.train_2019_df['type'] = 'new' # indicate data from new competition (2019)

        # create validation set from 2019

        self.train_2019_df, self.val_df = train_test_split(

            self.train_2019_df, test_size=0.1, stratify=self.train_2019_df.diagnosis, random_state=1337)

        # combine training set

        self.train_df = pd.concat([self.train_2015_df, self.train_2019_df], axis=0)

        self.init_model(load_pretrained, freeze_pretrained)

        self.best_score = None

        self.logger = NNLogger()

        

    def init_model(self, pretrained, freeze, num_classes=1):

        enet = EfficientNet.from_name('efficientnet-b0')

        if pretrained:

            enet.load_state_dict(torch.load('../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'))

        if freeze:

            for parameter in enet.parameters():

                parameter.requires_grad = True

        n_fc = enet._fc.in_features

        enet._fc = nn.Sequential(

                          nn.Dropout(p=0.5),

                          nn.Linear(in_features=n_fc, out_features=n_fc, bias=True),

                          nn.ReLU(),

                          nn.BatchNorm1d(n_fc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.5),

                          nn.Linear(in_features=n_fc, out_features=1, bias=True),

                         )

        self.model = OrdinalLogisticModel(enet, num_classes=5)

        self.model.to(device)

        

    def load_best_state_dict(self):

        self.model.load_state_dict(self.best_state_dict)

        

    def unfreeze(self):

        for parameter in self.model.parameters():

            parameter.requires_grad = True

    

    def freeze_except_fc(self):

        for _, child in trainer.model.named_children():

            for name, params in child.named_parameters():

                if '_fc.' not in name and name != 'cutpoints':

                    params.required_grad = False

                    

    def resample(self, df, class_dist):

        resample_df = []

        for label, n_sample in enumerate(class_dist):

            if len(df[df['diagnosis'] == label]) < n_sample:

                resample_df.append(df[df['diagnosis'] == label].sample(n=n_sample, replace=True))

            else:

                resample_df.append(df[df['diagnosis'] == label].sample(n=n_sample))

        return pd.concat(resample_df, axis=0)

                    

    def adjusted_weights(self, weights):

        return [weight * len(weights) / sum(weights) for weight in weights]

    

    def lr_finder(self, lr_find_epochs, start_lr, end_lr, train_type,

                  img_size, img_scale, train_transform, use_base_transform, batch_size, 

                  class_weights, num_workers=4):

        dataset = RetinaDataset(self.train_df[self.train_df['type'] == train_type],

                                img_size=img_size, img_scale=img_scale,

                                train_transform=train_transform,

                                use_base_transform=use_base_transform,

                                use_cache=self.use_cache)

        dataloader = DataLoader(dataset, batch_size=batch_size, 

                                shuffle=True, num_workers=num_workers)

        lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(dataloader)))

        param_list = [{"params": self.model.predictor._conv_stem.parameters(), "lr":start_lr * 0.1},

         {"params": self.model.predictor._bn0.parameters(), "lr":start_lr * 0.1},

         {"params": self.model.predictor._blocks.parameters(), "lr":start_lr * 0.1},

         {"params": self.model.predictor._conv_head.parameters(), "lr":start_lr * 0.1},

         {"params": self.model.predictor._bn1.parameters(), "lr":start_lr * 0.1},

         {"params": self.model.predictor._fc.parameters()},

         {"params": self.model.link.parameters()}]

        optimizer = torch.optim.Adam(param_list, lr=start_lr, weight_decay=1e-5)

        #optimizer = torch.optim.SGD(param_list, lr=start_lr, momentum=0.9, nesterov=True, weight_decay=1e-5)

        criterion = CumulativeLinkLoss(class_weights=self.adjusted_weights(class_weights))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Make lists to capture the logs

        lr_find_loss = []

        lr_find_lr = []

        iter = 0

        smoothing = 0.05

        for i in range(lr_find_epochs):

            for imgs, labels in tqdm(dataloader, total=len(dataloader), disable=(not DEBUG)):

                # Send to device

                imgs, labels = imgs.to(device), labels.to(device)

                # Training mode and zero gradients

                self.model.train()

                optimizer.zero_grad()

                # Get outputs to calc loss

                outputs = self.model(imgs)

                loss = criterion(outputs, labels)

                # Backward pass

                loss.backward()

                optimizer.step()

                # Update LR

                scheduler.step()

                lr_step = optimizer.state_dict()["param_groups"][-1]["lr"]

                lr_find_lr.append(lr_step)

                # smooth the loss

                if iter==0:

                    lr_find_loss.append(loss)

                else:

                    loss = smoothing  * loss + (1 - smoothing) * lr_find_loss[-1]

                    lr_find_loss.append(loss)

            iter += 1

        return lr_find_loss, lr_find_lr

    

    def cosine_lr(self, stepsize, min_lr, max_lr, lr_scale):

        

        def _cosine_lr(it):

            cycle_no = it // stepsize

            lr_scale_cycle = lr_scale ** cycle_no

            scaled_max_lr, scaled_min_lr = max_lr * lr_scale_cycle, min_lr * lr_scale_cycle

            fraction_to_restart = (it % stepsize) / stepsize

            lr = scaled_min_lr + 0.5 * (scaled_max_lr - scaled_min_lr) * (1 + np.cos(fraction_to_restart * np.pi))

            return lr

        

        return _cosine_lr

        

    def train_loop(self, n_epochs, img_size, img_scale, batch_size, class_weights, lr, lr_scale,

                   step_size, train_transform, use_base_transform, train_type,

                   best_score=None, save_model=True, num_workers=4):

        self.train_dataset = RetinaDataset(self.train_df[self.train_df['type'] == train_type], 

                                           img_size=img_size, img_scale=img_scale,

                                           train_transform=train_transform,

                                           use_base_transform=use_base_transform,

                                           use_cache=self.use_cache)

        self.val_dataset = RetinaDataset(self.val_df, img_size=img_size, img_scale=img_scale,

                                         train_transform=train_transform,

                                         use_base_transform=use_base_transform,

                                         use_cache=self.use_cache)

        self.train_dl = DataLoader(self.train_dataset, batch_size=batch_size, 

                                   shuffle=True, num_workers=num_workers)

        self.val_dl = DataLoader(self.val_dataset, batch_size=batch_size, 

                                 shuffle=True, num_workers=num_workers)

        self.criterion = CumulativeLinkLoss(class_weights=self.adjusted_weights(class_weights))

        self.best_score = best_score if best_score is not None else self.best_score

        epoch_lr_size = step_size * len(self.train_dl)

        param_list = [{"params": self.model.predictor._conv_stem.parameters()},

         {"params": self.model.predictor._bn0.parameters()},

         {"params": self.model.predictor._blocks.parameters()},

         {"params": self.model.predictor._conv_head.parameters()},

         {"params": self.model.predictor._bn1.parameters()},

         {"params": self.model.predictor._fc.parameters()},

         {"params": self.model.link.parameters()}]

        lr_fn = self.cosine_lr

        lr_params = [

            # _conv_stem

            lr_fn(epoch_lr_size, lr*0.01/6, lr*0.1, lr_scale=lr_scale),

            #_bn0

            lr_fn(epoch_lr_size, lr*0.01/6, lr*0.1, lr_scale=lr_scale),

            #_blocks

            lr_fn(epoch_lr_size, lr*0.1/6, lr*0.2, lr_scale=lr_scale),

            #_conv_head

            lr_fn(epoch_lr_size, lr*0.1/6, lr*0.2, lr_scale=lr_scale),

            #_bn1

            lr_fn(epoch_lr_size, lr*0.1/6, lr*0.2, lr_scale=lr_scale),

            #_fc

            lr_fn(epoch_lr_size, lr/6, lr, lr_scale=lr_scale),

            #link

            lr_fn(epoch_lr_size, lr/6, lr, lr_scale=lr_scale)

        ]

        self.optimizer = torch.optim.Adam(param_list, lr=1.)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_params)

        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1", verbosity=0)

        for epoch in range(n_epochs):

            print('Epoch: {}/{} \t LR: {}'.format(epoch + 1, n_epochs, self.scheduler.get_lr()[-1]))

            avg_loss = self.train_model()

            avg_val_loss = self.eval_model()

            train_kappa, val_kappa = self.logger.get_last_kappa_score()

            train_mae, val_mae = self.logger.get_last_mae()

            print('loss={:.4f} \t val_loss={:.4f}'.format(avg_loss, avg_val_loss))

            print('kappa={:.4f} \t val_kappa={:.4f}'.format(train_kappa, val_kappa))

            print('MAE={:.4f} \t val_MAE={:.4f}'.format(train_mae, val_mae))

            print('cutpoints={}'.format(str(self.model.link.cutpoints.detach().cpu().numpy())))

            print('Validation CM:')

            print(self.logger.get_last_cm())

            self.logger.step()

            if save_model and (self.best_score is None or val_kappa > self.best_score):

                print('best score improved to {:.4f}. Saving model.'.format(val_kappa))

                self.best_score = val_kappa

                self.best_state_dict = self.model.state_dict()

                save_score = str(int(self.best_score * 1000))

                torch.save(self.model.state_dict(), '../working/weight_best_kappa.pt')

        

    def train_model(self):

        self.model.train() 

        avg_loss = 0.

        self.optimizer.zero_grad()

        train_iter = tqdm(enumerate(self.train_dl), total=len(self.train_dl), disable=(not DEBUG))

        for idx, (imgs, labels) in train_iter:

            start_time = time.time()

            imgs_train, labels_train = imgs.to(device), labels.to(device)

            output_train = self.model(imgs_train)

            pred_diagnosis = output_train.detach().cpu().numpy().argmax(axis=1)

            loss = self.criterion(output_train, labels_train)

            self.logger.add('train', labels.numpy().ravel(), pred_diagnosis, loss.item())

            elapsed_time = time.time() - start_time

            self.logger.record_stat(self.scheduler.get_lr(), elapsed_time)

            train_iter.set_postfix(loss="{:.6f}".format(loss.item()))

#             with amp.scale_loss(loss, self.optimizer) as scaled_loss:

#                 scaled_loss.backward()

            loss.backward()

            self.optimizer.step() 

            self.scheduler.step()

            self.optimizer.zero_grad() 

            avg_loss += loss.item() / len(self.train_dl)

        return avg_loss

    

    def eval_model(self):

        self.model.eval()

        avg_val_loss = 0.

        with torch.no_grad():

            for idx, (imgs, labels) in tqdm(enumerate(self.val_dl), total=len(self.val_dl), disable=(not DEBUG)):

                imgs_val, labels_val = imgs.to(device), labels.to(device)

                output_val = self.model(imgs_val)

                val_diagnosis = output_val.detach().cpu().numpy().argmax(axis=1)

                loss = self.criterion(output_val, labels_val)

                self.logger.add('val', labels.numpy().ravel(), val_diagnosis, loss.item())

                avg_val_loss += loss.item() / len(self.val_dl)

        return avg_val_loss

    
IMG_SIZE = 224

train_params = {

    'n_epochs': 2,

    'img_size': IMG_SIZE,

    'img_scale': 1.2,

    'batch_size': 32,

    'class_weights': [1, 2, 1, 2, 1],

    'lr': 2e-3,

    'lr_scale': 0.8,

    'step_size': 1,

    'train_type': 'old',

    'use_base_transform' : ['weighted'],

    'train_transform': transforms.Compose([

            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),

            transforms.RandomRotation(90),

            transforms.ColorJitter(contrast=.1, saturation=.1, brightness=.1),

            transforms.RandomCrop(IMG_SIZE)

        ])

}
trainer = BlindnessDetectionTrainer(freeze_pretrained=False, use_cache=DEBUG)

lr_find_loss, lr_find_lr = trainer.lr_finder(lr_find_epochs=6, start_lr=3e-5, end_lr=3e-2, train_type='old',

                                             img_size=train_params['img_size'], img_scale=train_params['img_scale'],

                                             train_transform=train_params['train_transform'], 

                                             use_base_transform=train_params['use_base_transform'], batch_size=train_params['batch_size'], 

                                             class_weights=[1,2,1,2,1])

fig, ax = plt.subplots(1,1)

_ = ax.plot(lr_find_lr, lr_find_loss)

_ = ax.set_xscale('log')
# warm start by freezing pretrained layers

trainer = BlindnessDetectionTrainer(freeze_pretrained=True, use_cache=DEBUG)

trainer.train_loop(**train_params)
trainer.train_dataset.show_sample_imgs(6, get_original=True, use_train_transform=False)
trainer.train_dataset.show_sample_imgs(6, get_original=False, use_train_transform=train_params['train_transform'])
train_params['n_epochs'] = 4

train_params['lr'] = 3e-3

train_params['step_size'] = 2

train_params['train_type'] = 'old'

trainer.load_best_state_dict()

trainer.unfreeze()

trainer.train_loop(**train_params)
train_params['n_epochs'] = 2

train_params['step_size'] = 1

train_params['lr'] = 1e-3

train_params['train_type'] = 'new'

trainer.load_best_state_dict()

trainer.freeze_except_fc()

trainer.train_loop(**train_params)
trainer.train_dataset.show_sample_imgs(6, get_original=True, use_train_transform=False)
trainer.train_dataset.show_sample_imgs(6, get_original=False, use_train_transform=train_params['train_transform'])
train_params['n_epochs'] = 8

train_params['step_size'] = 2

train_params['use_base_transform'] = ['weighted']

trainer.load_best_state_dict()

trainer.unfreeze()

trainer.train_loop(**train_params)
_ = plt.plot([lr[-1] for lr in trainer.logger.lr_history])
_ = plt.plot([np.mean(loss) for loss in trainer.logger.loss['train']])

_ = plt.plot([np.mean(loss) for loss in trainer.logger.loss['val']])
class BlindnessDetectionPredictor(object):

    

    def __init__(self, subm_df, state_dict_file, transform, use_base_transform, img_size, 

                 img_scale, n_TTA, batch_size):

        self.subm_df = subm_df

        self.subm_df['img_path'] = self.subm_df['id_code'].apply(

            lambda f: _subm_2019_img_path + f + _subm_2019_ext if os.path.isfile(

                _subm_2019_img_path + f + _subm_2019_ext) else None)

        self.subm_df = self.subm_df.set_index('id_code')

        self.subm_dataset = RetinaDataset(self.subm_df, img_size=img_size, img_scale=img_scale,

                                          train_transform=transform, use_base_transform=use_base_transform)

        self.subm_dl = DataLoader(self.subm_dataset, batch_size=batch_size, shuffle=False)

        self.batch_size = batch_size

        self.n_TTA = n_TTA

        self.init_model(state_dict_file)

        

    def init_model(self, state_dict_file):

        enet = EfficientNet.from_name('efficientnet-b0')

        n_fc = enet._fc.in_features

        enet._fc = nn.Sequential(

                          nn.Dropout(p=0.5),

                          nn.Linear(in_features=n_fc, out_features=n_fc, bias=True),

                          nn.ReLU(),

                          nn.BatchNorm1d(n_fc, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=n_fc, out_features=1, bias=True),

                         )

        self.model = OrdinalLogisticModel(enet, num_classes=5)

        self.model.load_state_dict(torch.load(state_dict_file))

        self.model.to(device)

        

    def del_model(self):

        del self.model

        gc.collect()



    def predict(self):

        preds = np.zeros((len(self.subm_df), 5))

        self.model.eval()

        start_time = time.time()

        for tta_no in range(self.n_TTA):

            len_batch = len(self.subm_dl)

            with torch.no_grad():

                for i, data in enumerate(self.subm_dl):

                    images, _ = data

                    images = images.to(device)

                    output_subm = self.model(images)

                    preds[i * self.batch_size:(i + 1) * self.batch_size] += output_subm.detach().cpu().squeeze().numpy()

                    elapsed_time = time.time() - start_time

                    print('TTA={}/{}, Batch={}/{}, total_elapsed_time={:.2f}s'.format(

                        tta_no+1, self.n_TTA, i+1, len_batch, elapsed_time))

        output = preds / self.n_TTA

        self.del_model()

        return output
# submission set

_subm_2019_csv = '../input/aptos2019-blindness-detection/sample_submission.csv'

_subm_2019_img_path = '../input/aptos2019-blindness-detection/test_images/'

_subm_2019_ext = '.png'

submit_df = pd.read_csv(_subm_2019_csv)
IMG_SIZE = 224

subm_params = {

    'subm_df': submit_df,

    'state_dict_file' : '../working/weight_best_kappa.pt',

    'batch_size': 32,

    'n_TTA': 3,

    'img_size': IMG_SIZE,

    'img_scale': 1.2,

    'use_base_transform': ['crop', 'weighted'],

    'transform': transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(90),

        # no jitter for submission

        transforms.RandomCrop(IMG_SIZE)

    ])

}

enet_predictor = BlindnessDetectionPredictor(**subm_params)

enet_subm_preds = np.argmax(enet_predictor.predict(), axis=1)
submission = pd.DataFrame(

    {

        'id_code': submit_df.id_code.values,

        'diagnosis': enet_subm_preds

    }

)

print(submission.head())

print(submission.diagnosis.value_counts())

submission.to_csv('submission.csv', index=False)

print(os.listdir('./'))
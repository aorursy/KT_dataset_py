USE_COLAB, TRAIN_MODE = 0, 0
import os, shutil, sys, time, gc

import logging, copy, multiprocessing

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.auto import tqdm as tqdmauto

from PIL import Image

from collections import OrderedDict



import math, cv2, sklearn, albumentations

import torch

from torch import nn

from torch.nn import functional as F

from torch.utils import model_zoo

from torch.utils.data import Dataset, DataLoader, RandomSampler

from sklearn.model_selection import StratifiedKFold



if not USE_COLAB:

    print(os.listdir('../input/'))

    PATH = '../input/plant-pathology-2020-fgvc7/'

    SAVE_PATH = './'  # 模型要保存到的路径

else:

    from google.colab import drive



    drive.mount('/content/drive', force_remount=True)

    PATH = './drive/My Drive/Competition/plant-pathology-2020/plant-pathology-2020/'

    SAVE_PATH = PATH

    !git clone -q https://github.com/welkin-feng/ComputerVision.git

    sys.path.append('./ComputerVision/')

    !git clone -q https://github.com/rwightman/pytorch-image-models.git

    sys.path.append('./pytorch-image-models/')



print("PATH: ", os.listdir(PATH))

print("SAVE_PATH: ", os.listdir(SAVE_PATH))



# Gets the GPU if there is one, otherwise the cpu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

random_seed = 644

print(f'random_state: {random_seed}')
DEVICE = DEVICE

PATH = PATH

SAVE_PATH = SAVE_PATH



IMAGE_FILE_NAME = ['train_images_320x320.npy', 'test_images_320x320.npy']

IMG_SHAPE = (1365, 2048, 3)

INPUT_IMG_SHAPE = (320, 320, 3)

IMG_MEAN = np.array([0])

IMG_STD = np.array([1])



train_transforms = {

    'mix_prob': 1.0, 'mixup_prob': 0.2, 'cutmix_prob': 0.35, 'fmix_prob': 0, 

    'grid_prob': 0.2, 'erase_prob': 0, 'cutout_prob': 0, 

    'cutout_ratio': (0.1, 0.5), 'cut_size': int(INPUT_IMG_SHAPE[0] * 0.8), # (0.1, 0.3), 

    'brightness': (0.7, 1.1), 'noise_prob': 0, 'blur_prob': 0, 'drop_prob': 0, 'elastic_prob': 0,

    'hflip_prob': 0.1, 'vflip_prob': 0, 'scale': (0.8, 1.1), 

    'shear': (-10, 10), 'translate_percent': (-0.15, 0.15), 'rotate': (-20, 20)

}



n_fold = 5

fold = (0,) # (0, 1, 2, 3, 4)

BATCH_SIZE = 64

TEST_BATCH_SIZE = 16

accumulation_steps = 1

loss_weights = (1, 1)



learning_rate = 1e-3

lr_ratio = np.sqrt(0.1)

reduce_lr_metric = ['loss', 'score', 'both'][0]

patience = 5

num_classes = 4



n_epochs = 50

train_epochs = 50

resume = False

pretrained = not resume
df_train = pd.read_csv(PATH + 'train.csv')

df_train['class'] = np.argmax(df_train.iloc[:, 1:].values, axis=1)



skf = StratifiedKFold(n_fold, shuffle = True, random_state = 644)

for i_fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train['class'].values)):

    df_train.loc[val_idx, 'fold'] = i_fold

df_train['fold'] = df_train['fold'].astype(int)



df_test = pd.read_csv(PATH + 'test.csv')

submission = pd.read_csv(PATH + 'sample_submission.csv')
img_name = df_train['image_id'].iloc[0]

img = cv2.imread(PATH + 'images/{}.jpg'.format(img_name))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)

plt.show()

img = cv2.resize(img, (512, 320), interpolation = cv2.INTER_AREA).astype('uint8')

img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_CLOCKWISE)

print(img.shape)

plt.imshow(img)

plt.show()
train_data1 = np.zeros((len(df_train), 320, 320, 3), 'uint8')

train_data2 = np.zeros((len(df_train), 320, 512, 3), 'uint8')

for i in range(len(df_train)):

    img_name = df_train['image_id'].iloc[i]

    img = cv2.imread(PATH + 'images/{}.jpg'.format(img_name))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape != IMG_SHAPE:

        print(img_name)

        img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_CLOCKWISE)

    img1 = cv2.resize(img, (320, 320), interpolation = cv2.INTER_AREA).astype('uint8')

    train_data1[i] = img1

    img2 = cv2.resize(img, (512, 320), interpolation = cv2.INTER_AREA).astype('uint8')

    train_data2[i] = img2



np.save(SAVE_PATH + 'train_images_320x320.npy', train_data1)

np.save(SAVE_PATH + 'train_images_320x512.npy', train_data2)
del train_data1, train_data2

gc.collect()
test_data1 = np.zeros((len(df_test), 320, 320, 3), 'uint8')

test_data2 = np.zeros((len(df_test), 320, 512, 3), 'uint8')



for i in range(len(df_test)):

    img_name = df_test['image_id'].iloc[i]

    img = cv2.imread(PATH + 'images/{}.jpg'.format(img_name))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape != IMG_SHAPE:

        print(img_name)

        img = cv2.rotate(img, rotateCode = cv2.ROTATE_90_CLOCKWISE)

    img1 = cv2.resize(img, (320, 320), interpolation = cv2.INTER_AREA).astype('uint8')

    test_data1[i] = img1

    img2 = cv2.resize(img, (512, 320), interpolation = cv2.INTER_AREA).astype('uint8')

    test_data2[i] = img2



np.save(SAVE_PATH + 'test_images_320x320.npy', test_data1)

np.save(SAVE_PATH + 'test_images_320x512.npy', test_data2)
shutil.copy(PATH + 'train.csv', SAVE_PATH)

shutil.copy(PATH + 'test.csv', SAVE_PATH)

shutil.copy(PATH + 'sample_submission.csv', SAVE_PATH)
class PlantPathologyDataset(Dataset):

    def __init__(self, csv, idx, mode, transform = None, data = None):

        self.csv = csv.reset_index(drop = True)

        self.data = data

        self.filepath_format = PATH + 'images/{}.jpg'

        self.idx = np.asarray(idx).astype('int')

        self.mode = mode

        self.transform = transform



    def __len__(self):

        return self.idx.shape[0]



    def __getitem__(self, index):

        index = self.idx[index]

        if self.data is not None:

            image = self.data[index]

        else:

            img_name = self.csv['image_id'].iloc[index]

            image = cv2.imread(self.filepath_format.format(img_name))

        if image.shape != INPUT_IMG_SHAPE:

            image = cv2.resize(image, INPUT_IMG_SHAPE[:2], interpolation = cv2.INTER_AREA)



        image_origin = image.astype('float32').copy()

        image = self.transform(image).astype('float32') if self.transform is not None else image.astype('float32')

        image, image_origin =  np.rollaxis(image, 2, 0) / 255, np.rollaxis(image_origin, 2, 0) / 255



        if self.mode == 'test':

            return torch.tensor(image)

        else:

            label = self.csv.iloc[index, 1:5].values.astype('float32') # len = 4

            return torch.tensor(image), torch.tensor(image_origin), torch.tensor(label)



# transforms

transforms_train = lambda image: albumentations.Compose([

    albumentations.Cutout(max_h_size=train_transforms['cut_size'], max_w_size=train_transforms['cut_size'], num_holes=1, p=0.7),

])(image=image)['image']

transforms_val = None



def get_train_val_dataloader(i_fold):

    train_idx, valid_idx = np.where((df_train['fold'] != i_fold))[0], np.where((df_train['fold'] == i_fold))[0]

    train_data = np.load(SAVE_PATH + 'train_images_320x320.npy')

    dataset_train = PlantPathologyDataset(df_train, train_idx, 'train', transform=transforms_train, data = train_data)

    dataset_valid = PlantPathologyDataset(df_train, valid_idx, 'val', transform=transforms_val, data = train_data)

    train_loader = DataLoader(dataset_train, BATCH_SIZE, sampler=RandomSampler(dataset_train), num_workers=4, pin_memory=True)

    valid_loader = DataLoader(dataset_valid, TEST_BATCH_SIZE, sampler=None, num_workers=4, pin_memory=True)



    return train_loader, valid_loader



def get_test_dataloader():

    test_data = np.load(SAVE_PATH + 'test_images_320x320.npy')

    dataset_test = PlantPathologyDataset(df_test, np.arange(len(df_test)), 'test', data = test_data)

    test_loader = DataLoader(dataset_test, TEST_BATCH_SIZE, sampler=None, num_workers=4)



    return test_loader
df_show = df_train.iloc[:100]

dataset_show = PlantPathologyDataset(df_show, list(range(df_show.shape[0])), 'train', transform=None)



from pylab import rcParams

rcParams['figure.figsize'] = 20,10

for i in range(2):

    f, axarr = plt.subplots(1,5)

    for p in range(5):

        idx = np.random.randint(0, len(dataset_show))

        t0 = time.time()

        img, img_org, label = dataset_show[idx]

        # print(f"{time.time()-t0:.4f}")

        axarr[p].imshow(img.transpose(0, 1).transpose(1,2).squeeze())

        axarr[p].set_title(idx)
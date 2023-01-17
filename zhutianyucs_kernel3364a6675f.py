import sys

package_path = '../input/efficientnet/efficientnet-pytorch/EfficientNet-PyTorch/'

sys.path.append(package_path)
import os

import time

from functools import partial

import random



import cv2

import numpy as np

import pandas as pd

import scipy as sp

import torch

import torch.nn as nn

from efficientnet_pytorch import EfficientNet

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold

from torch.utils.data import Dataset

from torchvision import transforms

from tqdm import tqdm
class Config:

    train_img_dir = '../input/aptos2019-blindness-detection/train_images/'

    test_img_dir = '../input/aptos2019-blindness-detection/test_images/'



    processed_train_img_dir = '../data/aptos2019-blindness-detection/train_images/'

    process_test_img_dir = '../input/aptos2019-blindness-detection/test_images/'



    train_csv_path = '../input/aptos2019-blindness-detection/train.csv'

    test_csv_path = '../input/aptos2019-blindness-detection/test.csv'



    img_size = 256

    seed = 42

    

config = Config()
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    

def crop_image_from_gray(img, tol=7):

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1), mask.any(0))]

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol



        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]

        if (check_shape == 0):  # image is too dark so that we crop out everything,

            return img  # return original image

        else:

            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]

            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]

            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]

            #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1, img2, img3], axis=-1)

        #         print(img.shape)

        return img



def load_ben_color(path, img_size, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (img_size, img_size))

    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image
class MyDataset(Dataset):

    def __init__(self, dataframe, dirname, transform=None):

        self.df = dataframe

        self.dirname = dirname

        self.transform = transform



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        if 'diagnosis' not in self.df:

            label = 0

        else:

            label = self.df.diagnosis.values[idx]



        label = np.expand_dims(label, -1)

        img_id = self.df.id_code.values[idx]

        img_path = os.path.join(self.dirname, f'{img_id}.png')

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = crop_image_from_gray(image)

        image = cv2.resize(image, (256, 256))

        image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 30) ,-4 ,128)

        

        image = transforms.ToPILImage()(image)



        if self.transform:

            image = self.transform(image)



        return image, label



class OptimizedRounder(object):

    def __init__(self):

        self.coef_ = 0



    def _mae_loss(self, coef, X, y):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        score = eval_function(X_p, y)

        return -score



    def fit(self, X, y):

        loss_partial = partial(self._mae_loss, X=X, y=y)

        initial_coef = [0.5, 1.5, 2.5, 3.5]

        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')



    def predict(self, X, coef):

        X_p = np.copy(X)

        for i, pred in enumerate(X_p):

            if pred < coef[0]:

                X_p[i] = 0

            elif pred >= coef[0] and pred < coef[1]:

                X_p[i] = 1

            elif pred >= coef[1] and pred < coef[2]:

                X_p[i] = 2

            elif pred >= coef[2] and pred < coef[3]:

                X_p[i] = 3

            else:

                X_p[i] = 4

        return X_p



    def coefficients(self):

        return self.coef_['x']



def test_inference(model_name, ckpt_dir, coef, tta=10):

    seed_everything(config.seed)



    model = EfficientNet.from_name(model_name)

    in_features = model._fc.in_features

    model._fc = nn.Linear(in_features, 1)

    model.cuda()



    n_folds = sum([1 if name.endswith('.pt') else 0 for name in os.listdir(ckpt_dir)])



    train_transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation((-120, 120)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



    df = pd.read_csv(config.test_csv_path)

    print(f'test df len: {len(df)}')



    sorted_ckpts = sorted(os.listdir(ckpt_dir), key=lambda name: int(name.split('.')[0][4:]))



    tta_preds = []



    testset = MyDataset(df, config.process_test_img_dir, transform=train_transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4)

    print(f'testset len: {len(testset)}')



    for _tta in range(tta):

        for fold in range(n_folds):



            oof_preds = []

            ckpt_path = os.path.join(ckpt_dir, sorted_ckpts[fold])

            model.load_state_dict(torch.load(ckpt_path))

            print(f'Load checkpoint from {ckpt_path}')



            start_time = time.time()

            model.eval()

            with torch.no_grad():

                for idx, (imgs, labels) in enumerate(test_loader):

                    imgs_vaild, labels_vaild = imgs.cuda(), labels.float().cuda()

                    output_test = model(imgs_vaild)

                    oof_preds.append(output_test.squeeze(-1).cpu().numpy())

            elapsed_time = time.time() - start_time

            tta_preds.append(np.concatenate(oof_preds))

            print(f'TTA {_tta} Fold {fold + 1} time={elapsed_time:.2f}s')



    tta_preds = np.mean(tta_preds, axis=0)

    print(f'len tta_preds: {len(tta_preds)}')



    opt = OptimizedRounder()

    tta_preds = opt.predict(tta_preds, coef)



    return tta_preds



def make_submission(tta_preds, output_path):

    test_df = pd.read_csv(config.test_csv_path)

    sub = pd.DataFrame()

    print(len(test_df), len(tta_preds))

    sub['id_code'] = test_df.id_code

    sub['diagnosis'] = tta_preds.astype(np.int)

    sub.to_csv(output_path, index=False)

    print(f'saved in {output_path}')
! ls ../input/efficientnetb1/efficientnet-b1
tta_preds = test_inference('efficientnet-b1', '../input/efficientnetb1/efficientnet-b1', [0.50202154, 1.51818165, 2.87326943, 2.99355952], tta=1)

make_submission(tta_preds, 'submission.csv')
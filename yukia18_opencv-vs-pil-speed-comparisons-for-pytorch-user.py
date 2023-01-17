import os

import gc

import time



import cv2

import albumentations as A

import numpy as np

from albumentations.pytorch import ToTensor

from PIL import Image



import torch

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms





INPUT_DIR = '../input/all-dogs/all-dogs'
image_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR)]

print('total image files: {}'.format(len(image_files)))
def test(f, n_trials=5):

    elapsed_times = []

    for i in range(n_trials):

        t1 = time.time()

        f()

        t2 = time.time()

        elapsed_times.append(t2-t1)

    print('Mean: {:.3f}s - Std: {:.3f}s - Max: {:.3f}s - Min: {:.3f}s'.format(

        np.mean(elapsed_times),

        np.std(elapsed_times),

        np.max(elapsed_times),

        np.min(elapsed_times)

    ))
image_files_1000 = image_files[:1000]
def cv2_imread(path):

    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



f = lambda: [cv2_imread(f) for f in image_files_1000]
test(f, n_trials=5)
f = lambda: [Image.open(f) for f in image_files_1000]
test(f, n_trials=5)
# use them following test



cv2_img_1000 = [cv2_imread(f) for f in image_files_1000]

pil_img_1000 = [Image.open(f) for f in image_files_1000]
cv2_transform = A.Compose([A.Resize(64, 64, interpolation=cv2.INTER_LINEAR)])



f = lambda: [cv2_transform(image=img)['image'] for img in cv2_img_1000]
test(f, n_trials=5)
pil_transform = transforms.Compose([transforms.Resize((64, 64), interpolation=2)])



f = lambda: [pil_transform(img) for img in pil_img_1000]
test(f, n_trials=5)
# use them following test



cv2_img_1000 = [cv2_transform(image=img)['image'] for img in cv2_img_1000]

pil_img_1000 = [pil_transform(img) for img in pil_img_1000]
cv2_transform = A.Compose([ToTensor()])



f = lambda: [cv2_transform(image=img)['image'] for img in cv2_img_1000]
test(f, n_trials=5)
pil_transform = transforms.Compose([transforms.ToTensor()])



f = lambda: [pil_transform(img) for img in pil_img_1000]
test(f, n_trials=5)
image_files_1000 = image_files[:1000]
class BaseDataset(Dataset):

    def __init__(self, files, transform=None):

        super().__init__()

        self.files = files

        self.transform = transform

    

    def __len__(self):

        return len(self.files)





class PILDataset(BaseDataset):

    def __getitem__(self, idx):

        file = self.files[idx]

        img = Image.open(file)

        if self.transform is not None:

            img = self.transform(img)

            

        return img



    

class CV2Dataset(BaseDataset):

    def __getitem__(self, idx):

        file = self.files[idx]

        img = cv2.imread(file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:

            img = self.transform(image=img)['image']

            

        return img
def dataloader_test(files, transform, test_type='cv2', batch_size=64, n_trials=5):

    assert test_type in ['cv2', 'pil']

    

    if test_type == 'cv2':

        test_dataset = CV2Dataset(files, transform=transform)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    else:

        test_dataset = PILDataset(files, transform=transform)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    

    def f():

        for batch in test_dataloader:

            pass

    

    test(f, n_trials=n_trials)
cv2_transform = A.Compose([

    A.Resize(64, 64, interpolation=cv2.INTER_LINEAR),

    ToTensor()

])
dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)
pil_transform = transforms.Compose([

    transforms.Resize((64, 64), interpolation=2),

    transforms.ToTensor()

])
dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)
cv2_transform = A.Compose([

    A.SmallestMaxSize(64, interpolation=cv2.INTER_LINEAR),

    A.CenterCrop(64, 64),

    ToTensor()

])
dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)
pil_transform = transforms.Compose([

    transforms.Resize(64, interpolation=2),

    transforms.CenterCrop(64),

    transforms.ToTensor()

])
dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)
cv2_transform = A.Compose([

    A.Resize(64, 64, interpolation=cv2.INTER_LINEAR),

    A.HorizontalFlip(p=0.5),

    ToTensor()

])
dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)
pil_transform = transforms.Compose([

    transforms.Resize((64, 64), interpolation=2),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor()

])
dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)
cv2_transform = A.Compose([

    A.Resize(96, 96, interpolation=cv2.INTER_LINEAR),

    A.RandomCrop(64, 64),

    ToTensor()

])
dataloader_test(image_files_1000, cv2_transform, test_type='cv2', batch_size=64, n_trials=5)
pil_transform = transforms.Compose([

    transforms.Resize((96, 96), interpolation=2),

    transforms.RandomCrop(64),

    transforms.ToTensor()

])
dataloader_test(image_files_1000, pil_transform, test_type='pil', batch_size=64, n_trials=5)
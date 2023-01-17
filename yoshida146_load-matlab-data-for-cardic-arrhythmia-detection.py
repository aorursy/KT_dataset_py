import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import joblib

from tqdm.notebook import tqdm

from scipy.io import loadmat

from glob import glob



sns.set()

sns.set_context('poster')

%matplotlib inline
# Setting Random Seed



import random

import tensorflow as tf

import numpy as np

import os



def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    

seed_everything(seed=42)
_input_path = os.path.join('..', 'input', '1056lab-cardiac-arrhythmia-detection')

os.listdir(_input_path)
af_files = sorted(glob(os.path.join(_input_path, 'af', '*.mat')))

af_files[: 10]
normal_files = sorted(glob(os.path.join(_input_path, 'normal', '*.mat')))

normal_files[: 10]
test_files = sorted(glob(os.path.join(_input_path, 'test', '*.mat')))

test_files[: 10]
def check_folder(fold_name, search_ext='mat', path='../input/1056lab-cardiac-arrhythmia-detection'):

    """ フォルダ内のファイル拡張子がひとつだけか確認する """

    flag = False

#     extension = '*.{}'.format(search_ext)

    for f in sorted(os.listdir(os.path.join(path, fold_name))):

        if not f.endswith('.{}'.format(search_ext)):

            print(f)

            flag = True

    if not flag:

        print('{} Folder is only *.{}.'.format(fold_name, search_ext))

    

    return not flag
check_folder('normal')

check_folder('af')

check_folder('test')
def load_data(pathes, label=None, prefix=None, max_length=0, min_length=np.inf, verbose=True):

    verbose = not verbose # tqdm用に反転

    

    if prefix is not None:

        if prefix.endswith("/") or prefix.endswith("\\"):

            prefix += "//"



        for i, val in enumerate(pathes):

            pathes[i] = "{}{}".format(prefix, val)

    

    data_array = []

    labels = []

    for i, f in enumerate(tqdm(pathes, disable=verbose)):

        tmp = loadmat(f)

        data_array.append(tmp["val"].flatten())

        tmp_len = len(data_array[i])

        

        if max_length < tmp_len:

            max_length = tmp_len

        elif min_length > tmp_len:

            min_length = tmp_len

            

        if label is not None:

            labels.append(label)

        

    return data_array, labels, max_length, min_length
normals, normal_label, max_length, min_length = load_data(normal_files, label=0)

afs, af_label, max_length, min_length = load_data(af_files, label=1, max_length=max_length, min_length=min_length)

tests, _, max_length, min_length = load_data(test_files, label=None, max_length=max_length, min_length=min_length)

labels = np.append(normal_label, af_label)
print("Max Length : {}\nMin Length : {}".format(max_length, min_length))
normals_ = normals.copy()

for i, v in enumerate(tqdm(normals_)):

    normals_[i] = v[: min_length]



afs_ = afs.copy()

for i, v in enumerate(tqdm(afs_)):

    afs_[i] = v[: min_length]



tests_ = tests.copy()

for i, v in enumerate(tqdm(tests_)):

    tests_[i] = v[: min_length]
normals_ = np.array(normals_).reshape(-1, min_length, 1)

afs_ = np.array(afs_).reshape(-1, min_length, 1)

tests_ = np.array(tests_).reshape(-1, min_length, 1)
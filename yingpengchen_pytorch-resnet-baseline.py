import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import torch

import torch.nn as nn

import multiprocessing

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

MIN_SAMPLES_PER_CLASS = 100

BATCH_SIZE = 64

NUM_WORKERS = multiprocessing.cpu_count()

NUM_TOP_PREDICTS = 1

ENABLE_FAST_SKIP = False
train = pd.read_csv('../input/landmark-recognition-2020/train.csv')

test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

train_dir = '../input/landmark-recognition-2020/train/'

test_dir = '../input/landmark-recognition-2020/test/'
import time

import sys

sys.path.append('../input/landmarkrecognition2020')

from datasets.datasets import *

import models
def load_data(train, test, train_dir, test_dir):

    counts = train.landmark_id.value_counts()

    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index

    num_classes = selected_classes.shape[0]

    print('classes with at least N samples:', num_classes)



    train_mask = train.landmark_id.isin(selected_classes)

    train = train.loc[train_mask].copy()

    print('train_df', train.shape)

    print('test_df', test.shape)



    # filter non-existing test images

    exists = lambda img: os.path.exists(f'{test_dir}/{img[0]}/{img[1]}/{img[2]}/{img}.jpg')

    test_mask = test.id.apply(exists)

    test = test.loc[test_mask].copy()

    print('test_df after filtering', test.shape)



    label_encoder = LabelEncoder()

    label_encoder.fit(train.landmark_id.values)

    print('found classes', len(label_encoder.classes_))

    assert len(label_encoder.classes_) == num_classes



    train.landmark_id = label_encoder.transform(train.landmark_id)



    train_dataset = ImageDataset(train, train_dir, mode='train')

    test_dataset = ImageDataset(test, test_dir, mode='test')



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,

                              shuffle=False, num_workers=4, drop_last=True)



    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,

                             shuffle=False, num_workers=NUM_WORKERS)



    return train_loader, test_loader, label_encoder, num_classes
def inference(data_loader, model):

    model.eval()



    activation = nn.Softmax(dim=1)

    all_predicts, all_confs, all_targets = [], [], []



    with torch.no_grad():

        for i, data in enumerate(tqdm(data_loader)):

            if data_loader.dataset.mode != 'test':

                input_, target = data['image'], data['target']

            else:

                input_, target = data['image'], None



            output = model(input_.cuda())

            output = activation(output)



            confs, predicts = torch.topk(output, NUM_TOP_PREDICTS)

            all_confs.append(confs)

            all_predicts.append(predicts)



            if target is not None:

                all_targets.append(target)

            

#             if (i+1)%10 == 0:

#                 print('Have processed images is ', (i+1)*BATCH_SIZE)



    predicts = torch.cat(all_predicts)

    confs = torch.cat(all_confs)

    targets = torch.cat(all_targets) if len(all_targets) else None



    return predicts, confs, targets
def generate_submission(test_loader, model, label_encoder):

    sample_sub = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')



    predicts_gpu, confs_gpu, _ = inference(test_loader, model)

    predicts, confs = predicts_gpu.cpu().numpy(), confs_gpu.cpu().numpy()



    labels = [label_encoder.inverse_transform(pred) for pred in predicts]

    print('labels')

    print(np.array(labels))

    print('confs')

    print(np.array(confs))



    sub = test_loader.dataset.df

    def concat(label: np.ndarray, conf: np.ndarray) -> str:

        return ' '.join([f'{L} {c}' for L, c in zip(label, conf)])

    sub['landmarks'] = [concat(label, conf) for label, conf in zip(labels, confs)]



    sample_sub = sample_sub.set_index('id')

    sub = sub.set_index('id')

    sample_sub.update(sub)



    sample_sub.to_csv('submission.csv')

    print(sub)
if __name__ == '__main__':

    global_start_time = time.time()

    train_loader, test_loader, label_encoder, num_classes = load_data(train, test, train_dir, test_dir)

    arch = 'resnet50'

    model_path = '../input/modelspath/resnet50_best.pth.tar'



#     if ENABLE_FAST_SKIP and test.id[0] == "00084cdf8f600d00":

    # This is a run on the public data, skip it to speed up submission run on private data.

    print("Skipping run on public test set.")

    sample_sub = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

    sample_sub.to_csv('submission.csv')

#     else:

#         #create model

#         print("==> creating model '{}'".format(arch))

#         model = models.__dict__[arch](num_classes = num_classes)

#         model.cuda()



#         #original saved file with DataParallel or without DataParallel

#         state_dict = torch.load(model_path)['state_dict']

#         #create new OrderedDict that does not contain 'module.'

#         from collections import OrderedDict

#         new_state_dict = OrderedDict()

#         for k,v in state_dict.items():

#             if 'module' in k:

#                 k = k[7:] # remove 'module.'

#             new_state_dict[k] = v

#         #load params

#         model.load_state_dict(new_state_dict)

#         model.eval()



#         print('inference mode')

#         generate_submission(test_loader, model, label_encoder)
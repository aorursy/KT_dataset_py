import numpy as np

import os

import torch



from torch import nn, optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from typing import Dict



from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import AgentDataset, EgoDataset

from l5kit.rasterization import build_rasterizer



from matplotlib import pyplot as plt
DIR_INPUT = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"



SINGLE_MODE_SUBMISSION = f"{DIR_INPUT}/single_mode_sample_submission.csv"

MULTI_MODE_SUBMISSION = f"{DIR_INPUT}/multi_mode_sample_submission.csv"
cfg = {

    'format_version': 4,

    'model_params': {

        'model_architecture': 'resnet50',

        'history_num_frames': 10,

        'history_step_size': 1,

        'history_delta_time': 0.1,

        'future_num_frames': 50,

        'future_step_size': 1,

        'future_delta_time': 0.1

    },

    

    'raster_params': {

        'raster_size': [224, 224],

        'pixel_size': [0.5, 0.5],

        'ego_center': [0.25, 0.5],

        'map_type': 'py_semantic',

        'satellite_map_key': 'aerial_map/aerial_map.png',

        'semantic_map_key': 'semantic_map/semantic_map.pb',

        'dataset_meta_key': 'meta.json',

        'filter_agents_threshold': 0.5

    },

    

    'train_data_loader': {

        'key': 'scenes/train.zarr',

        'batch_size': 32,

        'shuffle': True,

        'num_workers': 4

    },

    

    'val_data_loader': {

        'key': 'scenes/validate.zarr',

        'batch_size': 32,

        'shuffle': False,

        'num_workers': 4

    },

    



}
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT

dm = LocalDataManager(None)
train_cfg = cfg["train_data_loader"]

val_cfg = cfg["val_data_loader"]



# Rasterizer

rasterizer = build_rasterizer(cfg, dm)



# Train dataset/dataloader

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset,

                              shuffle=train_cfg["shuffle"],

                              batch_size=train_cfg["batch_size"],

                              num_workers=train_cfg["num_workers"])
train_dataset[0]['image'].shape
plt.figure(figsize=(48,48))

for i in range(11):

    plt.subplot(11, 1, i+1)

    plt.imshow(train_dataset[0]['image'][i])

plt.show()
plt.figure(figsize=(48,48))

for i in range(11):

    plt.subplot(11, 1, i+1)

    plt.imshow(train_dataset[0]['image'][i+11])

plt.show()
plt.figure(figsize=(12,12))

for i in range(3):

    plt.subplot(3, 1, i+1)

    plt.imshow(train_dataset[0]['image'][i+22])

plt.show()
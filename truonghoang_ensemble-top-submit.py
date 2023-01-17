import numpy as np 

import pandas as pd 

import os
os.listdir('../input/')
dsub = [  # all 0.978

#     pd.read_csv('../input/plantpathology/effnet-fastai-folds-x5_version3.csv'),

    pd.read_csv('../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv'),  # ok

    pd.read_csv('../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv'),  # okok

    pd.read_csv('../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv'),  # ok

#     pd.read_csv('../input/plantpathology/tpu-ensemble-effnb7-effnb6-inceptresnetv2-etc_verion13.csv'),

    pd.read_csv('../input/plantpathology/classification-densenet201-efficientnetb7.csv'),  # ok

#     pd.read_csv('../input/plantpathology/plant-pathology-2020-efficientnetb7-0-980-score.csv'),

    pd.read_csv('../input/plantpathology/tf-zoo-models-on-tpu.csv'),  # ok

#     pd.read_csv('../input/plantpathology/plant-pathology-2020-in-pytorch-0-979-score_version12.csv'),

]



# Increase the weight for the importance file by add them again, again ...

pd.read_csv('../input/plantpathology/tf-zoo-models-on-tpu.csv')

dsub.append(pd.read_csv('../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv'))

dsub.append(pd.read_csv('../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv'))



n = len(dsub)
sub = pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')

sub.head()
sub.healthy = 0

sub.multiple_diseases = 0

sub.rust = 0

sub.scab = 0



for d in dsub:

    sub.healthy += d.healthy

    sub.multiple_diseases += d.multiple_diseases

    sub.rust += d.rust

    sub.scab += d.scab



sub.healthy = sub.healthy/n

sub.multiple_diseases = sub.multiple_diseases/n

sub.rust = sub.rust/n

sub.scab = sub.scab/n



sub.to_csv('submission.csv', index=False)

sub.head()
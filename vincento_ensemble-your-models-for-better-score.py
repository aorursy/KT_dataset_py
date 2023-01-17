from __future__ import division

from glob import glob

import sys

import pandas as pd

import numpy as np

import os
glob_files = './s*.csv'

root_folder = '/Users/red/Desktop/geo_mean/'

outfile_name = 'kaggle_mean_ensemble.csv'

dataframes = '{}'
glob_files = './s*.csv'

root_folder = '/Users/red/Desktop/geo_mean/'

outfile_name = 'kaggle_mean_ensemble.csv'

dataframes = '{}'
images = dataframes['./s1.csv']['image']

COLUMNS = dataframes['./s5.csv'].columns
KEYS = dataframes.keys()

images = dataframes['./s1.csv']['image']

f_submit = open(os.path.join(root_folder, outfile_name), 'wa')

f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')

for image in images:

    BET = []

    ALB = []

    DOL =[]

    LAG = []

    NoF = []

    OTHER = []

    SHARK = []

    YFT = []

    for key in KEYS:

        row = dataframes[key][dataframes[key]['image'] == image]

        BET.append(row['BET'].values)

        ALB.append(row['ALB'].values)

        DOL.append(row['DOL'].values)

        LAG.append(row['LAG'].values)

        NoF.append(row['NoF'].values)

        OTHER.append(row['OTHER'].values)

        SHARK.append(row['SHARK'].values)

        YFT.append(row['YFT'].values)

    

    write_row = ",".join([image,str(np.mean(ALB)),str(np.mean(BET)),str(np.mean(DOL)),str(np.mean(LAG)),str(np.mean(NoF)),str(np.mean(OTHER)),str(np.mean(SHARK)),str(np.mean(YFT))]) 

    f_submit.write(write_row)

    f_submit.write("\n")

f_submit.close()
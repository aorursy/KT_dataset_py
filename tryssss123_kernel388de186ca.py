# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

structures = pd.read_csv('../input/structures.csv')
from torchvision import transforms, utils

from torch.utils.data import Dataset, DataLoader



import random

import numbers

import math

import collections



import numpy as np

import torch
# should add CV

class MolecularDataset(Dataset):

    def __init__(self, df, structures, mode="train", transform="augmentation"):

        self.mode = mode

        self.transform = transform

        self.df = df

        self.structures = structures

#         self.df.reset_index(drop=True)

#         self.structures.reset_index(drop=True)

        self.types = ['1JHC', '2JHH', '1JHN', '2JHN', '2JHC', '3JHH', '3JHC', '3JHN']

        

    def __len__(self):

        if self.mode == "train":

            return 4000000

        else:

            return 658147

    

    def points_to_3D_map(self, points):

        '''points: n*4(x,y,z,index) '''

        grid = np.zeros((256,256,256))

        for point in points:

#             print(point)

            x = int(point[0]*10)+128

            y = int(point[1]*10)+128

            z = int(point[2]*10)+128

            grid[x][y][z] = point[3]

        return grid

    

    def type_to_num(self, type):

        num = 0

        for i in range(8):

            if type == self.types[i]:

                num = i

                break

        return num

            

            

#         index                                      0

# id                                         0    [0]

# molecule_name               dsgdb9nsd_000001    [1]

# atom_index_0                               1    [2]

# atom_index_1                               0

# type                                    1JHC

# scalar_coupling_constant             84.8076

# atom_count                                 5

# Name: 0, dtype: object

    def __getitem__(self, idx):

        if self.mode == "train":

            infomation = np.zeros((3,))

#             print("**xxxxxxxxx")

#             print(self.df.loc[idx][2])

#             print(self.df.loc[idx][3])

#             print(self.df.loc[idx][4])

            infomation[0] = self.df.loc[idx]['atom_index_0']

            infomation[1] = self.df.loc[idx]['atom_index_1']

            infomation[2] = self.type_to_num(self.df.loc[idx]['type'])

            

            name = self.df.loc[idx]['molecule_name']

#             print("here 3")

            structures_of_name = structures.loc[structures['molecule_name'] == name]

#             print("here 4")

            points = np.zeros((structures_of_name.shape[0], 4))

            for i in range(structures_of_name.shape[0]):

                points[i, 0] = structures_of_name.loc[i]['x']

                points[i, 1] = structures_of_name.loc[i]['y']

                points[i, 2] = structures_of_name.loc[i]['z']

                points[i, 3] = structures_of_name.loc[i]['atom_index']

#             print("here 5")

#             grid = self.points_to_3D_map(points)

            

            target = np.array(self.df.loc[idx]['scalar_coupling_constant'])

            return torch.from_numpy(infomation), torch.from_numpy(points), torch.from_numpy(target)

        else:

            infomation = np.zeros((3,))

            infomation[0] = self.df.loc[idx]['atom_index_0']

            infomation[1] = self.df.loc[idx]['atom_index_1']

            infomation[2] = self.type_to_num(self.df.loc[idx]['type'])

            

            name = self.df.loc[idx][1]

            structures_of_name = structures.loc[structures['molecule_name'] == name]

            points = np.zeros((structures_of_name.shape[0], 4))

            for i in range(structures_of_name.shape[0]):

                points[i, 0] = structures_of_name.loc[i]['x']

                points[i, 1] = structures_of_name.loc[i]['y']

                points[i, 2] = structures_of_name.loc[i]['z']

                points[i, 3] = structures_of_name.loc[i]['atom_index']

    

#             grid = self.points_to_3D_map(points)

            

            target = np.array(self.df.loc[idx]['scalar_coupling_constant'])

            return torch.from_numpy(infomation), torch.from_numpy(points), torch.from_numpy(target)
dataloader = MolecularDataset(df=train_df, structures=structures, mode='train')

for i in range(len(dataloader)):

    information, grid, target = dataloader[i]

    print("*****in *****")

    print(information)

    print("*****grid*****")

    print(grid)

    print("*****target*****")

    print(target)

    if i == 0:

        break
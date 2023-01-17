# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sub1 = pd.read_csv('../input/smiles/vgg_face.csv')

sub2 = pd.read_csv('../input/smiles/vgg_face (1).csv')

sub3 = pd.read_csv('../input/smiles/vgg_face (2).csv')

sub4 = pd.read_csv('../input/smiles/vgg_face (3).csv')

sub5 = pd.read_csv('../input/smiles/sub_detect_kinship.csv')

temp=pd.read_csv('../input/smiles/vgg_face (1).csv')
sns.set(rc={'figure.figsize':(18,6.5)})

sns.kdeplot(sub1['is_related'],label="sub1",shade=True,bw=.1)

sns.kdeplot(sub2['is_related'], label="sub2",shade=True,bw=.1)

sns.kdeplot(sub3['is_related'], label="sub3",shade=True,bw=.1)

sns.kdeplot(sub4['is_related'], label="sub4",shade=True,bw=.1)

sns.kdeplot(sub5['is_related'], label="sub5",shade=True,bw=.1)
sns.set()

plt.hist(sub1['is_related'],bins=50)

plt.show()
sns.set()

plt.hist(sub1['is_related'],bins=50)

plt.show()
sns.set()

plt.hist(sub2['is_related'],bins=50)

plt.show()
sns.set()

plt.hist(sub3['is_related'],bins=50)

plt.show()
sns.set()

plt.hist(sub4['is_related'],bins=50)

plt.show()
sns.set()

plt.hist(sub5['is_related'],bins=50)

plt.show()
temp['is_related'] = 0.18*sub1['is_related'] + 0.16*sub2['is_related'] + 0.22*sub3['is_related'] + 0.22*sub4['is_related'] + 0.22*sub4['is_related'] 

temp.to_csv('submission1.csv', index=False )
sns.set()

sns.kdeplot(sub1['is_related'],label="sub1",shade=True,bw=.1)

plt.show()
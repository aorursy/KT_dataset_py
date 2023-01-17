import os

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
sub1 = pd.read_csv('../input/smiles/vgg_face.csv')

sub2 = pd.read_csv('../input/smiles/vgg_face (1).csv')

sub3 = pd.read_csv('../input/smiles/submission (1).csv')

temp=pd.read_csv('../input/smiles/submission (1).csv')

sns.set(rc={'figure.figsize':(18,6.5)})

sns.kdeplot(sub1['is_related'],label="sub1",shade=True,bw=.1)

sns.kdeplot(sub2['is_related'], label="sub2",shade=True,bw=.1)

sns.kdeplot(sub3['is_related'], label="sub3",shade=True,bw=.1)

temp['is_related'] = 0.60*sub1['is_related'] + 0.22*sub2['is_related'] + 0.18*sub3['is_related'] 

temp.to_csv('submission4.csv', index=False )
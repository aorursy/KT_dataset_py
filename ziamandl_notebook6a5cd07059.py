# A histogram

import numpy as np 

import pandas as pd 

train = pd.read_csv("../input/train.csv")

train.head()
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4), dpi=100)
train.revenue.hist()

train.hist()

#Revenue histrogram is close to normally distributed and slightly right-tailed.  

#There are multiple hitrograms shown with P1, P2, P3,... data.  

#These histrograms are not normally distributed

train.City.scatter()
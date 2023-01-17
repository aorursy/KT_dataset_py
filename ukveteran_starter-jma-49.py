from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/Lottery_NY_Lotto_Winning_Numbers__Beginning_2001.csv')
dat.columns = ['Date', 'Numbers', 'Bonus','Extra']

dat["AllNumbers"] = dat["Numbers"].map(str) + " " + dat["Bonus"].map(str)

df = dat.copy()

del df['Bonus']

del df['Extra']

del df['Numbers']
df2 = pd.DataFrame(df['AllNumbers'].str.split(" ").apply(pd.Series, 0).stack())

df2.index = df2.index.droplevel(-1)

df2.head(20)
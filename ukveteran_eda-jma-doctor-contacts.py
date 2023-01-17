from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/contacts-with-medical-doctors/DoctorContacts.csv')

dat.head()
sns.distplot(dat["ndisease"])
sns.scatterplot(x='age',y='ndisease',data=dat)
sns.countplot(dat["ndisease"])
sns.countplot(dat["sex"])
sns.countplot(dat["child"])
sns.countplot(dat["black"])
sns.countplot(dat["health"])
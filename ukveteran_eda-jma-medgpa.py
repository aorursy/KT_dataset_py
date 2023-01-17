from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import os 

import pandas as pd



dat = pd.read_csv('../input/gpa-and-medical-school-admission/MedGPA.csv')

dat.head()
sns.distplot(dat["GPA"])
sns.distplot(dat["MCAT"])
sns.scatterplot(x='GPA',y='MCAT',data=dat)
sns.countplot(dat["GPA"])
sns.countplot(dat["MCAT"])
sns.countplot(dat["Sex"])
sns.countplot(dat["Accept"])
sns.countplot(dat["Apps"])
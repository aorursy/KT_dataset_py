import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import random
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
train_features.head(5)
train_features.describe()
train_features.dtypes
test_features.head(5)
test_features.describe()
train_features.isnull().sum()
test_features.isnull().sum()
train_targets_scored.isnull().sum()
train_targets_nonscored.isnull().sum()
train_features.columns.str.startswith('g-').sum()
train_features.columns.str.startswith('c-').sum()
print('Number of rows : ', train_targets_scored.shape[0])
print('Number of cols : ', train_targets_scored.shape[1])
train_targets_scored.head()

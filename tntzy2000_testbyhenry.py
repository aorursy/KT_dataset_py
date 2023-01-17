import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv(r'../input/train.csv')
data.head()
sns.boxplot(x='Pclass', y='Age', data=data);
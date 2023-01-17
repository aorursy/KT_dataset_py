# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
data_white = pd.read_csv("../input/winequality-white.csv")
data_red = pd.read_csv("../input/winequality-red.csv")

data_white.head(5)
data_white.info()
data_white.describe()
data_white.shape
data_white.columns
plt.figure()
data_white.plot(kind='box',by='quality')
plt.title('Box Plot By Quality')
ax = data_white.plot(kind='box')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.title('Box Plot By Quality')
ax = data_white['alcohol'].plot(kind='box')
plt.setp(ax.get_xticklabels(), rotation=45)
plt.title('Box Plot By Quality')
data_white.boxplot('alcohol',by ='quality')









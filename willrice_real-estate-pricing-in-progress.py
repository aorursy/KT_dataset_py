# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import test and train datasets

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

correlationMatrix = train.corr(method='pearson', min_periods=1)

correlationMatrix
sns.heatmap(correlationMatrix)
train.plot.scatter(x='LotArea', y='SalePrice')

train.plot.scatter(x='GarageArea', y='GarageCars')

train.plot.scatter(x='1stFlrSF', y='SalePrice')
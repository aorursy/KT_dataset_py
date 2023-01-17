# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

train_csv = pd.read_csv('../input/train.csv')
test_csv = pd.read_csv('../input/test.csv')
train_csv[:5]
test_csv[:5]
train_csv.describe()
train_csv_length = len(train_csv)
train_csv_length
x = train_csv['x']
y = train_csv['y']

from matplotlib.colors import LogNorm

bins = 20
while bins <=160:
    plt.hist2d(x, y, bins=bins, norm=LogNorm())
    plt.colorbar()
    plt.title('x and y location histogram - ' + str(bins) + ' bins')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    bins = bins * 2
len(train_csv['place_id'].unique())
train_place_distribution = train_csv['place_id'].value_counts().tolist()
plt.title('Histogram of place_id frequency')
plt.xlabel('place_id frequency')
plt.ylabel('Frequency')
_, _, _ = plt.hist(train_place_distribution, 50, normed=1, facecolor='green', alpha=0.75)
train_norm = (train_csv - train_csv.mean()) / (train_csv.max() - train_csv.min())
print (np.correlate(train_norm['x'], train_csv['place_id']))
print (np.correlate(train_norm['y'], train_csv['place_id']))
corr_matrix = train_norm.corr(method='pearson', min_periods=1)
print (corr_matrix)

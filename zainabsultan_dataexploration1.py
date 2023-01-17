#imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#load the data
train = pd.read_csv('../input//forest-cover-type-kernels-only/train.csv.zip')
test = pd.read_csv('../input//forest-cover-type-kernels-only/train.csv.zip')

#getting the first five rows can give us a first impression of the data
train.head()
train.info()

train.describe()
#here we can see the mean and std of the data which gives an idea about the distribution
#we plot a pearson correlarion heat map
#the heat map provided by seaborn can give us an idea about the relations between the features and labels
corr = train.corr()
f, ax = plt.subplots(figsize=(35, 35))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})




z = np.abs(stats.zscore(train))

print(len(np.unique(np.where(z >= 3)[0])) )
#check for class imbalance
y = train.iloc[:,-1]
print(y.value_counts())
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bike = pd.read_csv("/kaggle/input/bids-machine-learning/train.csv")

bike.head()
bike.describe()
bike.isnull().sum()
import seaborn as sns



sns.distplot(bike.cnt)
sns.scatterplot(x=bike.id, y=bike.cnt)
sns.lineplot(x=bike.id, y=bike.cnt)
sns.boxplot(y=bike.cnt,x=bike.mnth)
sns.boxplot(y=bike.cnt,x=bike.season)
sns.boxplot(y=bike.cnt,x=bike.weekday)
sns.boxplot(y=bike.cnt,x=bike.holiday)
sns.boxplot(y=bike.cnt,x=bike.workingday)
sns.boxplot(y=bike.cnt,x=bike.weathersit)
sns.scatterplot(x=bike.windspeed, y=bike.cnt)
sns.scatterplot(x=bike.hum, y=bike.cnt)
sns.scatterplot(x=bike.temp, y=bike.cnt)
sns.scatterplot(x=bike.atemp, y=bike.cnt)
sns.scatterplot(x=bike.atemp, y=bike.temp)
g = sns.FacetGrid(bike, row="workingday", col="season", margin_titles=True)

g.map(sns.distplot,'cnt')
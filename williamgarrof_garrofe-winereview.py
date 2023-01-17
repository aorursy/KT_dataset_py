# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.model_selection as sk
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
ds= pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
ds.head(10)
ds.columns
from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
display(HTML(ds.head(10).to_html()))
ds.shape
ds.describe()
ds.columns
ds.points.count()
ds.points.value_counts()
ds.points.describe()
ds.country
ds.country == 'France'
ds.country + " - " + ds.region_1
ds['province'].value_counts().head(10).plot.bar()
d = ds[ds.variety.isin(ds.variety.value_counts().head(5).index)]

sns.boxplot(
    x='variety',
    y='points',
    data=d
)
ds['variety'].value_counts().head(15).plot.bar()
ds['points'].value_counts().sort_index().plot.bar()
sns.kdeplot(ds.query('price < 200').price)

ds.count()
ds[np.isnan(ds.price)]
ds = ds.fillna(value = {'price':ds.price.mean()})
plt.hist(ds.price, bins=80)
ds.info()
dsPred = ds[[ 'description', 'variety', 'province' ]]
dsPred.head()
ds1 = dsPred.sample(10000, random_state=1).copy()
X = ds1['description']
y = ds1['variety']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state = 0)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
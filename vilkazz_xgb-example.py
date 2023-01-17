# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_squared_error

from math import sqrt



import xgboost as xgb



import matplotlib.pyplot as plt#plot drawing

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# train_data = pd.read_csv('../input/2016.csv', header=0)

data = pd.read_csv('../input/2017.csv', header=0)

data.head()
print('Correlation Map')

data.corr() 

#correlation map view

f, ax = plt.subplots(figsize = (8,8))

sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt = ".2f", ax=ax)

plt.show()

data.describe()
from bubbly.bubbly import bubbleplot 

from __future__ import division

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



figure = bubbleplot(dataset=data, x_column='Happiness.Score', y_column='Happiness.Rank', 

    bubble_column='Country', size_column='Economy..GDP.per.Capita.', color_column='Country', 

    x_title="Happiness score", y_title="Happyness", title='Test bubbleplot',

    x_logscale=True, scale_bubble=3, height=650)



iplot(figure)
dts = data.drop(['Country', 'Happiness.Rank', 'Happiness.Score', 'Whisker.high', 'Whisker.low'], axis=1)

msk = np.random.rand(len(dts)) < 0.8

train = dts[msk]

test = dts[~msk]



labels = data[msk]['Happiness.Score']/10

test_labels = data[~msk]['Happiness.Score']/10
eval_set = [(train, labels), (test, test_labels)]

eval_metric = ["logloss", "rmse"]
dtrain = xgb.DMatrix(train.values, labels.values)
# specify parameters via map

param = {'max_depth':1, 'eta':2, 'silent':1, 'objective':'reg:linear', 'learning_rate': 0.1, 'n_estimators': 100 }

clf = xgb.XGBModel(**param)

clf.fit(train, labels,

        eval_set=eval_set,

        eval_metric=eval_metric,

        verbose=True

       )



evals_result = clf.evals_result()

# bst = xgb.train(param, dtrain, num_round)

# make prediction

# preds = bst.predict(dtest)
feature_df = pd.DataFrame(

    {'features': list(dts),

     'importances': clf.feature_importances_

    })



sns.set(style="whitegrid")



plot = sns.barplot(x="features", y="importances", data=feature_df)

plot.set_xticklabels(list(dts), rotation=30)
sqrt(mean_squared_error(test_labels, clf.predict(test)))


# dts.head()
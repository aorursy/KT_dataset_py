# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
bike = pd.read_csv("../input/train.csv")
bike.head()
bike.describe()
import seaborn as sns

sns.distplot(bike['cnt'])
g = sns.FacetGrid(bike, row="workingday", col="holiday", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')
g = sns.FacetGrid(bike, row="workingday", col="season", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')
g = sns.FacetGrid(bike, row="weathersit", col="season", margin_titles=True)
#g.map(ptl.hist,bike['cnt'])
g.map(sns.distplot,'cnt')
X_train = bike.drop(columns="cnt")
y_train = bike['cnt']

from sklearn.linear_model import LogisticRegression

# Initialize the predictive model object
mod_logistic = LogisticRegression()

# Train the model using the training sets
mod_logistic.fit(X_train, y_train)


# Make predictions using the testing set
pred = mod_logistic.predict(X_train)

sns.distplot(pred)
sns.scatterplot(pred, y_train)
test = pd.read_csv("../input/test.csv")
test.head()
test['cnt'] = mod_logistic.predict(test)
test.head()
res = test[(['id','cnt'])]
res.head()
### run this to generate the prediction file. Change each time the name by adding info related to the model and the version,
### you must upload this files into the kaggle platform (on our competition page) so this prediction can enter the challenge.

# res.to_csv("prediction_lm_v1.0.csv")
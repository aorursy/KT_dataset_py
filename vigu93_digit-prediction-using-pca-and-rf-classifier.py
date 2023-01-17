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
sourceData = pd.read_csv("../input/train.csv")
sourceData.head()
sourceData.describe()
y = sourceData.loc[:,'label']
y
X = sourceData.drop('label',axis=1)
X.head()
X.shape
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
grid = GridSearchCV(
    Pipeline([
        ('reduce_dim', PCA(random_state=100)),
        ('classify', RandomForestClassifier())
        ]),
    param_grid=[
        {
            'reduce_dim__n_components': range(200,400,50),
        }
    ],
    cv=5, scoring='accuracy')

grid.fit(X,y)
grid.best_estimator_
grid.best_params_
grid.best_score_
grid.cv_results_
pca_optimal = PCA(n_components=200,random_state=100)
X_train = pca_optimal.fit_transform(X)
rf = RandomForestClassifier()
rf.fit(X_train,y)
test = pd.read_csv("../input/test.csv")
test.head()
y_pred_PCA = rf.predict(pca_optimal.transform(test))
y_pred_PCA
data_to_submit = pd.DataFrame({
    'Label':y_pred_PCA[:]
})

data_to_submit = data_to_submit.reset_index().rename(columns={'index': 'ImageId', 'Label': 'Label'})
data_to_submit['ImageId'] += 1
data_to_submit.to_csv('csv_to_submit.csv',index=False)
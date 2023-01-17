# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
csv_train=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/train.csv', index_col=0)

csv_test=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/test_data.csv', index_col=0)

csv_submission=pd.read_csv('/kaggle/input/logistic-classification-diabetes-knn/submission_form.csv')



sns.distplot(csv_train["0"])

sns.distplot(csv_train["1"])

sns.distplot(csv_train["3"])

sns.distplot(csv_train["4"])

sns.distplot(csv_train["5"])

sns.distplot(csv_train["6"])

sns.distplot(csv_train["7"])
sns.pairplot(csv_train, hue='8')
from sklearn.neighbors import KNeighborsClassifier



x=csv_train.drop('8', axis=1)

y=csv_train['8']

test_x=csv_test.drop('8', axis=1)

knn=KNeighborsClassifier(metric='mahalanobis', metric_params={'V':np.cov(x, rowvar=False)})

knn.fit(x, y)

test_predict_y=knn.predict(test_x)
for idx, y in enumerate(test_predict_y):

    csv_submission['Label'][idx]=int(y)

    

csv_submission.dtypes
csv_submission=csv_submission.astype({'Label':int})
csv_submission.to_csv('submission.csv',index=False)
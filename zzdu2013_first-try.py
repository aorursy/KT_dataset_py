# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder

from sklearn import cross_validation

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
# load data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

ID = test_data['id']
contfeatures=[]

for c,x in train_data.iloc[1,:].iteritems():

    if (not str(x).isalpha()):

        contfeatures.append(c)

contfeatures_test=[]

for c,x in test_data.iloc[1,:].iteritems():

    if (not str(x).isalpha()):

        contfeatures_test.append(c)
contfeatures.remove('id')

contfeatures.remove('loss')

contfeatures_test.remove('id')
contfeatures.append('loss')
catfeatures=[]

for c,x in train_data.iloc[1,:].iteritems():

    if  str(x).isalpha():

        catfeatures.append(c)



for cf1 in catfeatures:

    le = LabelEncoder()

    le.fit(train_data[cf1].unique())

    train_data[cf1] = le.transform(train_data[cf1])
catfeatures_test=[]

for c,x in test_data.iloc[1,:].iteritems():

    if  str(x).isalpha():

        catfeatures_test.append(c)



for cf1 in catfeatures_test:

    le = LabelEncoder()

    le.fit(test_data[cf1].unique())

    test_data[cf1] = le.transform(test_data[cf1])
test_data.head()
X = np.array(train_data.drop(['id','loss'],1))

y = np.log1p(train_data['loss'])
linreg = LinearRegression()

linreg.fit(X, y)

y_pred = np.expm1(linreg.predict(test_data.drop(['id'],1)))
submission = pd.DataFrame({

        "id": ID,

        "loss": y_pred

    })

submission.to_csv('try.csv', index=False)
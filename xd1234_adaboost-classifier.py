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
import pandas as pd



df = pd.read_csv('../input/HR_comma_sep.csv')

df.describe()
df.info()
target = df['left']

df = df.drop(labels= ['left', 'sales'], axis = 1)
from sklearn.preprocessing import StandardScaler, LabelEncoder,MultiLabelBinarizer

from sklearn.cross_validation import train_test_split

import numpy as np

df['salary_encoder'] = LabelEncoder().fit_transform(df['salary'])

#print(k.shape)

#print(df.shape)

#df = encoder.transform(df)

#df = df + k

df = df.drop(labels = ['salary'], axis = 1)

#print (df)

scaler = StandardScaler().fit(df)

df = scaler.transform(df)



print(df)
x_train, x_test, y_train, y_test = train_test_split(df, target, test_size = 0.25, random_state = 0)
from sklearn.linear_model import Lasso

from sklearn.ensemble import AdaBoostClassifier



#lr = Lasso(alpha = 0.1).fit(x_train, y_train)

lr = AdaBoostClassifier(n_estimators = 100).fit(x_train, y_train)

lr.score(x_test, y_test)
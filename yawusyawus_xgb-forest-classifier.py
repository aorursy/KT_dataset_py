# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.metrics import accuracy_score



from xgboost import XGBClassifier, plot_importance



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')



y = train['Cover_Type'] # this is the target

X = train.drop('Cover_Type', axis = 1)

X_test = test.copy()



print('Train set shape : ', X.shape)

print('Test set shape : ', X_test.shape)
X.head()
X_test.head()
print('Missing Label? ', y.isnull().any())

print('Missing train data? ', X.isnull().any().any())

print('Missing test data? ', X_test.isnull().any().any())
print (X.dtypes.value_counts())

print (X_test.dtypes.value_counts())
X.describe()
X.nunique()
X.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)

X_test.drop(['Soil_Type15', 'Soil_Type7'], axis=1, inplace = True)
X_test.describe()
columns = X.columns
X_test_index = X_test.index # the scaler drops table index/columns and outputs simple arrays..

scaler = RobustScaler()

X = scaler.fit_transform(X)

X_test = scaler.transform(X_test)
X_train,  X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
xgb= XGBClassifier( n_estimators=1000,  #todo : search for good parameters

                    learning_rate= 0.5,  #todo : search for good parameters

                    objective= 'binary:logistic', #this outputs probability,not one/zero. should we use binary:hinge? is it better for the learning phase?

                    random_state= 1,

                    n_jobs=-1)
xgb.fit(X=X_train, y=y_train,

        eval_metric='merror', # merror: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases). 

        eval_set=[(X_val,y_val)],

        early_stopping_rounds = 100,

        verbose = False

       )

print(xgb.best_score)
plt.figure(figsize=(25,10))

sns.barplot(y=xgb.feature_importances_, x=columns)
xgb.fit(X,y)

preds_test = xgb.predict(X_test)

preds_test.shape
preds_test
# Save test predictions to file

output = pd.DataFrame({'ID': X_test_index,

                       'TARGET': preds_test})

output.to_csv('submission.csv', index=False)

output.head()
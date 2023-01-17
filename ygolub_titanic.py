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


df = pd.read_csv('../input/train.csv')

df_test= pd.read_csv('../input/test.csv')

df['Sex'] = df['Sex'].map({'male':0,'female':1})

df_test['Sex'] = df_test['Sex'].map({'male':0,'female':1})

#df.head()
ohe = pd.get_dummies(df['Embarked'])

df = pd.concat([df,ohe],axis = 1)



ohe_t = pd.get_dummies(df_test['Embarked'])

df_test = pd.concat([df_test,ohe],axis = 1)



X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','C','Q','S']]

X_test = df_test[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','C','Q','S']]

X.head(n=10)

Y = df['Survived']
from sklearn.preprocessing import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

imr = imr.fit(X)

imputed_data = imr.transform(X.values)



imr = imr.fit(X_test)

imputed_data_test = imr.transform(X_test.values)



from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(imputed_data)

#X_train_test =[imputed_data_test[:,0], stdsc.fit_transform(imputed_data_test[:,1:])]

X_train_test = stdsc.fit_transform(imputed_data_test[:,1:])

passids = imputed_data_test[:,0]

print(X_train_test)
from sklearn.svm import SVC

svm = SVC(kernel='rbf',  random_state=0, gamma=10, C=1000.0)

svm.fit(X_train_std, Y)

print('Training accuracy:', svm.score(X_train_std, Y))



result = np.column_stack((passids.T,svm.predict(X_train_test)))





dataframe = pd.DataFrame(data=result,    # values

             columns = ['PassengerId','Survived'])


dataframe.to_csv('result.csv',index=False)
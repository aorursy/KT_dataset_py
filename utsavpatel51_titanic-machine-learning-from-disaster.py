import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder

from sklearn.metrics import accuracy_score
dataset = pd.read_csv('../input/train.csv')

test_dataset= pd.read_csv('../input/test.csv')

len(dataset)
dataset.isnull().sum()
dataset = dataset.dropna(subset=['Embarked'])

dataset.Age.fillna(dataset.Age.mean(),inplace=True)

dataset.isnull().sum(),print(len(dataset))
le = LabelEncoder()

dataset['Sex']= le.fit_transform(dataset['Sex'])

dataset['Embarked']= le.fit_transform(dataset['Embarked'])



one_hot_sex = pd.get_dummies(dataset['Sex'],prefix='sex')

one_hot_embarked = pd.get_dummies(dataset['Embarked'],prefix='embarked')

ont_hot_pclass= pd.get_dummies(dataset['Pclass'],prefix='pclass')



dataset = dataset.drop('Sex',axis=1)

dataset = dataset.drop('Embarked',axis=1)

dataset = dataset.drop('Pclass',axis=1)



dataset = dataset.join(one_hot_sex)

dataset = dataset.join(one_hot_embarked)

dataset = dataset.join(ont_hot_pclass)
dataset.head()
scaler = MinMaxScaler()

dataset[['Age','Fare']] = scaler.fit_transform(dataset[['Age','Fare']])
X = dataset[['pclass_1','pclass_2','pclass_3','sex_0','sex_1','Age','SibSp','Parch','Fare','embarked_0','embarked_1','embarked_2']]

y = dataset['Survived']
#MAKE TEST DATASET READY
len(test_dataset)
test_dataset.isnull().sum()
#test_datasetdataset = test_dataset.dropna(subset=['Embarked'])

test_dataset.Age.fillna(test_dataset.Age.mean(),inplace=True)

test_dataset.Fare.fillna(test_dataset.Fare.mean(),inplace=True)

test_dataset.isnull().sum(),print(len(test_dataset))
le = LabelEncoder()

test_dataset['Sex']= le.fit_transform(test_dataset['Sex'])

test_dataset['Embarked']= le.fit_transform(test_dataset['Embarked'])



one_hot_sex = pd.get_dummies(test_dataset['Sex'],prefix='sex')

one_hot_embarked = pd.get_dummies(test_dataset['Embarked'],prefix='embarked')

ont_hot_pclass= pd.get_dummies(test_dataset['Pclass'],prefix='pclass')



test_dataset = test_dataset.drop('Sex',axis=1)

test_dataset = test_dataset.drop('Embarked',axis=1)

test_dataset = test_dataset.drop('Pclass',axis=1)



test_dataset = test_dataset.join(one_hot_sex)

test_dataset = test_dataset.join(one_hot_embarked)

test_dataset = test_dataset.join(ont_hot_pclass)
scaler = MinMaxScaler()

test_dataset[['Age','Fare']] = scaler.fit_transform(test_dataset[['Age','Fare']])
X_test = test_dataset[['pclass_1','pclass_2','pclass_3','sex_0','sex_1','Age','SibSp','Parch','Fare','embarked_0','embarked_1','embarked_2']]

len(X_test)
Pid = test_dataset['PassengerId']
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "output_1.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

# create a link to download the dataframe
#LOGISTIC DOWNLOAD

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1,solver='liblinear',penalty='l2')

lr.fit(X,y)

pre_x = lr.predict(X_test)

pre_x_df = pd.DataFrame({'Survived':pre_x},index=Pid)

create_download_link(pre_x_df)
#Training score

accuracy_score(y,lr.predict(X))
#SVM DOWNLOAD

from sklearn import svm

clf = svm.SVC(C=30,kernel='rbf',gamma='auto')

clf.fit(X,y)

pre_x_svm = clf.predict(X_test)

pre_x_svm_df = pd.DataFrame({'Survived':pre_x_svm},index=Pid)

create_download_link(pre_x_svm_df,filename='output_svm_c_30_scale.csv')
#Training score

accuracy_score(y,clf.predict(X))
#DecisionTree Download

from sklearn.tree import DecisionTreeClassifier

tree_ = DecisionTreeClassifier()

tree_.fit(X,y)

pre_x_dt = tree_.predict(X_test)

pre_x_dt_df = pd.DataFrame({'Survived':pre_x_dt},index=Pid)

create_download_link(pre_x_dt_df,filename='output_dt.csv')
#Training score

accuracy_score(y,tree_.predict(X))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X,y)

pre_x_knn = knn.predict(X_test)

pre_x_knn_df = pd.DataFrame({'Survived':pre_x_knn},index=Pid)

create_download_link(pre_x_knn_df,filename='output_knn_5.csv')
#Training score

accuracy_score(y,knn.predict(X))
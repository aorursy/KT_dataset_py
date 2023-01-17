# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 999)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

import warnings

warnings.filterwarnings("ignore")

from sklearn import preprocessing

from sklearn import model_selection

from sklearn import linear_model

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/bank-marketing-analysis/bank-additional-full.csv",sep=';')
train_data.head()
train_data.info()
train_data.describe()
train_data.isnull().sum()
train_data.duplicated().sum()
rcParams['figure.figsize'] = 17,10

sns.countplot(x=train_data['job'])
rcParams['figure.figsize'] = 17,10

sns.countplot(x=train_data['job'],hue=train_data['y'],palette="Set2")
sns.countplot(x=train_data['education'])
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['education'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['marital'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['housing'],hue=train_data['y'],palette="Set2")
train_data['loan'].value_counts().plot(kind="bar")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['loan'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['contact'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['nr.employed'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['poutcome'],hue=train_data['y'],palette="Set2")
train_data['month'].value_counts().plot(kind="pie")
rcParams['figure.figsize'] = 15,10

sns.countplot(x=train_data['month'],hue=train_data['y'],palette="Set2")
train_data["loan"].value_counts()
train_data["education"].value_counts().plot(kind="bar")
train_data["housing"].value_counts().plot(kind="pie")
train_data["contact"].value_counts().plot(kind="bar")
train_data['marital'].value_counts().plot(kind="pie")
train_data['campaign'].value_counts()
train_data['campaign'].value_counts().plot(kind="bar")
rcParams['figure.figsize'] = 15,10

sns.countplot(train_data['campaign'],hue=train_data['y'],palette="Set2")
rcParams['figure.figsize'] = 15,10

sns.countplot(train_data['pdays'],hue=train_data['y'],palette="Set2")
train_data['cons.price.idx'].value_counts().plot(kind="bar")
train_data.head()
new_df = train_data.copy(deep=True)
le = preprocessing.LabelEncoder()



# job

le.fit(new_df['job'])

new_df['job'] = le.transform(new_df['job'])



# maritial feature

le.fit(new_df['marital'])

new_df['marital'] = le.transform(new_df['marital'])



# education_feature

le.fit(new_df['education'])

new_df['education'] = le.transform(new_df['education'])



# housing_feature

le.fit(new_df['housing'])

new_df['housing'] = le.transform(new_df['housing'])



# loan_feature

le.fit(new_df['loan'])

new_df['loan'] = le.transform(new_df['loan'])



# contact_feature

le.fit(new_df['contact'])

new_df['contact'] = le.transform(new_df['contact'])



# Month_feature

le.fit(new_df['month'])

new_df['month'] = le.transform(new_df['month'])



# day of week_feature

le.fit(new_df['day_of_week'])

new_df['day_of_week'] = le.transform(new_df['day_of_week'])



# poutcome_feature

le.fit(new_df['poutcome'])

new_df['poutcome'] = le.transform(new_df['poutcome'])



# default_feature

le.fit(new_df['default'])

new_df['default'] = le.transform(new_df['default'])







# Target_feature

le.fit(new_df['y'])

new_df['y'] = le.transform(new_df['y'])



correleation_matrix = new_df.corr()
rcParams['figure.figsize'] = 25,20

sns.heatmap(correleation_matrix, cbar=True, square= True,fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')

y = new_df['y']

x = new_df.drop(['y'],axis=1)
X_train,X_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.10, random_state=42)
LR = linear_model.LogisticRegression()
LR.fit(X_train,y_train)
y_pred = LR.predict(X_test)
from sklearn import metrics

f1 = metrics.f1_score(y_true=y_test,y_pred=y_pred)

acc = metrics.accuracy_score(y_true=y_test,y_pred=y_pred)

pres = metrics.precision_score(y_true=y_test,y_pred=y_pred)

recall = metrics.recall_score(y_true=y_test,y_pred=y_pred)
print("The accuracy of the model Logistic Regression Model",acc)

print("The F1 Score of the model Logistic Regression Model",f1)

print("The Precision of the model Logistic Regression Model",pres)

print("The recall of the model Logistic Regression Model",recall)
from sklearn import ensemble

RFC = ensemble.RandomForestClassifier()
RFC.fit(X_train,y_train)
y_pred_rfc = RFC.predict(X_test)
f1_rfc = metrics.f1_score(y_true=y_test,y_pred=y_pred_rfc)

acc_rfc = metrics.accuracy_score(y_true=y_test,y_pred=y_pred_rfc)

pres_rfc = metrics.precision_score(y_true=y_test,y_pred=y_pred_rfc)

recall_rfc = metrics.recall_score(y_true=y_test,y_pred=y_pred_rfc)

cfn_matrix = metrics.plot_confusion_matrix(RFC,X_test,y_test)
print("The accuracy of the model RandomForestClassifier Model",acc_rfc)

print("The F1 Score of the model RandomForestClassifier Model",f1_rfc)

print("The Precision of the model RandomForestClassifier Model",pres_rfc)

print("The recall of the model RandomForestClassifier",recall_rfc)
ETC = ensemble.ExtraTreesClassifier()
ETC.fit(X_train,y_train)
y_pred_ETC = ETC.predict(X_test)
f1_ETC = metrics.f1_score(y_true=y_test,y_pred=y_pred_ETC)

acc_ETC = metrics.accuracy_score(y_true=y_test,y_pred=y_pred_ETC)

pres_ETC = metrics.precision_score(y_true=y_test,y_pred=y_pred_ETC)

recall_ETC = metrics.recall_score(y_true=y_test,y_pred=y_pred_ETC)
print("The accuracy of the modelExtraTreesClassifier Model",acc_ETC)

print("The F1 Score of the model ExtraTreesClassifier Model",f1_ETC)

print("The Precision of the modelExtraTreesClassifier Model",pres_ETC)

print("The recall of the model ExtraTreesClassifier Model",recall_ETC)
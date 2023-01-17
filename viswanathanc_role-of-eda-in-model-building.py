# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns #Data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings 

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing modules and data

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model.ridge import RidgeClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble.forest import ExtraTreesClassifier

from sklearn.ensemble.forest import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

# here we have nine models



from sklearn.metrics import accuracy_score,confusion_matrix



train_data=pd.read_csv('../input/learn-together/train.csv',index_col='Id')

test_data=pd.read_csv('../input/learn-together/test.csv',index_col='Id')
from sklearn.model_selection import train_test_split

X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y)

mod_val=[]
#Logistic Regression

model=LogisticRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Logistics Regression:',mod_val[0])
#Ridge Classifier

model=RidgeClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Ridge Classifier:',mod_val[1])
#SupportVectorMachine

#It takes a long time and also gives poor results

model=SVC()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Support Vector Classifier:',mod_val[2])
#Decision tree Regression

model=DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Decision Tree Classifier:',mod_val[3])
#GradientBoostingClassifier

model=GradientBoostingClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Gradient Boosting classifier:',mod_val[4])
#Adaboost classifier

model=AdaBoostClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Ada boost classifier:',mod_val[5])
#ExtraTree classifier

model=ExtraTreesClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Extra Tree classifier',mod_val[6])
#RandomForest Classifier

model=RandomForestClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Random Forest classifier:',mod_val[7])
#K nearest Neighbour classifier

model=KNeighborsClassifier()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

mod_val.append(round(accuracy,4))

print('Accuracy of Nearest neighbour:',mod_val[8])
train_data.describe().T
print(train_data.info())
print(train_data.head().T)
# Distribution of Data according to Forest cover type

values=train_data.Cover_Type.value_counts()

plt.bar(values.index,values)

plt.xlabel('Cover types')

plt.ylabel('Counts')

print(values)
#Dropping two columns

train_data.drop(['Soil_Type7','Soil_Type15'],axis=1,inplace=True)

train_data.shape
# Checking the two features

print(train_data.Soil_Type8.value_counts())

print(train_data.Soil_Type25.value_counts())
# Dropping the other two columns as well

train_data.drop(['Soil_Type8','Soil_Type25'],axis=1,inplace=True)

train_data.shape
# box plot with the 10 featured column that were noted earlier

Featured=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 

'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

plt.figure(figsize=(15,30))

for i,feature in enumerate(Featured):

    plt.subplot(5,2,i+1)

    sns.boxplot(x='Cover_Type',y=feature,data=train_data).set(title='{} vs Cover_Type'.format(feature))

plt.tight_layout()
#correlation matrix for featured columns vs others

Feature_Target=pd.concat([train_data[Featured],train_data[['Cover_Type']]],axis=1)

Feature_Target.columns
corr=Feature_Target.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr,square=True,center=0,linewidths=0.5,annot=True)

plt.title('Correlation Matrix')
#Wilderness

train_data.groupby('Cover_Type')['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4'].sum()
#Soil type. We have 36 soil types now and one cover_type. 

#so in total the last 37 soil types can be used for analyzing soil type

temp=train_data.iloc[:,-37:]

temp.groupby('Cover_Type').sum().T
# changing dtype of coulmn

for col in train_data:

    if col[:9]=='Soil_Type' or col[:10]=='Wilderness':

        train_data[col]=train_data[col].astype('category')

train_data['Cover_Type']=train_data['Cover_Type'].astype('category')

train_data.info()
X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y)
def model_build_evaluate(classifier,r_list):

    model=classifier()

    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)

    accuracy=accuracy_score(y_pred,y_test)

    r_list.append(round(accuracy,4))

    print('Accuracy of',classifier, ':',round(accuracy,4))

    
#list of classifiers

Classifiers=[LogisticRegression,RidgeClassifier,SVC,DecisionTreeClassifier,

             GradientBoostingClassifier,AdaBoostClassifier,ExtraTreesClassifier,

             RandomForestClassifier,KNeighborsClassifier]

eda_val=[]

for classifier in Classifiers:

    model_build_evaluate(classifier,eda_val)
ax=sns.scatterplot(x=list(range(9)),y=mod_val)

ax=sns.scatterplot(x=list(range(9)),y=eda_val)

plt.xticks(list(range(9)),Classifiers,rotation=90)

plt.legend(['With out EDA','After EDA'])
#Multinomial Naive Bayes is imported. It is more suited for discrete variables

from sklearn.naive_bayes import MultinomialNB
# Naive bayes classifier on categorical data

cat_data=train_data.iloc[:,-41:]

print(cat_data.columns)

X,y=cat_data.iloc[:,:-1],cat_data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y)

model=MultinomialNB()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_pred,y_test)

print('Accuracy of Multinomial Naive Bayes classifier',accuracy)
#Preprocessing

from sklearn.preprocessing import StandardScaler

#Now we have to make use of the whole model

X,y=train_data.iloc[:,:-1],train_data.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y)

#scaling the first 10 features alone

X_train[Featured]= StandardScaler().fit_transform(X_train[Featured])

X_test[Featured] = StandardScaler().fit_transform(X_test[Featured])
# classifying again

pp_val=[]

for classifier in Classifiers:

    model_build_evaluate(classifier,pp_val)
ax=sns.scatterplot(x=list(range(9)),y=mod_val)

ax=sns.scatterplot(x=list(range(9)),y=eda_val)

ax=sns.scatterplot(x=list(range(9)),y=pp_val)

plt.xticks(list(range(9)),Classifiers,rotation=90)

plt.legend(['With out EDA','After EDA','After Preprocessing'])
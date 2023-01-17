# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
#Visualization 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import missingno
import cufflinks as cf
import plotly.offline as po
import plotly.express as px
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import warnings
warnings.filterwarnings(action='once')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv') # Loading training Data
test_data = pd.read_csv('/kaggle/input/titanic/test.csv') # Loading Testing Data
## First 5 rows of training dataset
train_data.head(5)
## First 5 rows of testing dataset
test_data.head(5)
## last 5 rows of training dataset
train_data.tail(5)
## Last 5 rows of test dataset
test_data.tail(5)
## Data types of Each Attributes
train_data.dtypes
## Data Describe which gives mean max min 25% 50% and 75% answers of Each Feature
train_data.describe()
train_data['PassengerId'].isnull().sum() ## Getting total null values in PassengerID feature.
test_data['PassengerId'].isnull().sum() ## Getting total null values in PassengerID feature. Testing Data
train_data['Survived'].isnull().sum() ## Getting total null values in Survived feature.
train_data['Pclass'].isnull().sum() ## Getting total null values in Pclass feature.
test_data['Pclass'].isnull().sum() ## Getting total null values in Pclass feature.
train_data['Sex'].isnull().sum() ## Getting total null values in Sex feature.
test_data['Sex'].isnull().sum() ## Getting total null values in Sex feature.
train_data['Age'].isnull().sum() ## Getting total null values in Age feature.
test_data['Age'].isnull().sum() ## Getting total null values in Age feature.
mean = train_data['Age'].mean()
train_data['Age'] = train_data['Age'].replace(np.NAN,mean) ### Here Missing Values are been replaced with Mean value
mean = test_data['Age'].mean()
test_data['Age'] = test_data['Age'].replace(np.NAN,mean) ### Here Missing Values are been replaced with Mean value
train_data['SibSp'].isnull().sum() ## Getting total null values in SibSp feature.
test_data['SibSp'].isnull().sum() ## Getting total null values in SibSp feature of Testing Data.
train_data['Parch'].isnull().sum() ## Getting total null values in Parch feature.
test_data['Parch'].isnull().sum() ## Getting total null values in Parch feature testing Data.
train_data['Cabin'].isnull().sum() ## Getting total null values in Cabin feature.
test_data['Cabin'].isnull().sum() ## Getting total null values in Cabin feature of testing Data.
train_data['Embarked'].isnull().sum() ## Getting total null values in Embarked feature.
test_data['Embarked'].isnull().sum() ## Getting total null values in Embarked feature of Testing Data.
train_data['Cabin'].value_counts().to_frame()
test_data['Cabin'].value_counts().to_frame()
train_data['Embarked'].value_counts().to_frame()
new_train_data = train_data.copy()
new_train_data.head()
new_train_data.drop('Name',axis=1,inplace=True)
new_train_data['Embarked'] = new_train_data['Embarked'].replace(np.NAN,"S")
new_train_data['Embarked'].isnull().sum()
new_train_data['Sex'] = pd.get_dummies(new_train_data['Sex']) #### One Hot Encoder for Sex male =0 female = 1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lb = LabelEncoder()
new_train_data['Embarked'] = lb.fit_transform(new_train_data['Embarked'])
onehotencode = OneHotEncoder(handle_unknown='ignore')
encf = pd.DataFrame(onehotencode.fit_transform(new_train_data[['Embarked']]).toarray())
new_train_data = new_train_data.join(encf)
new_train_data.head()
## Rename the Columns 0,1,2 to C,Q,S
new_train_data.rename(columns={0:"C",1:"Q",2:"S"},inplace=True)
new_train_data.head()
fig=px.box(new_train_data,x='Fare')
fig.show()
fig = px.box(test_data,x='Fare')
fig.show()
fig = px.box(new_train_data,x='Age')
fig.show()
fig = px.box(test_data,x='Age')
fig.show()
new_train_data = new_train_data[new_train_data.Fare > 0]
new_train_data = new_train_data[new_train_data.Fare <= 300]
fig = px.box(new_train_data,x='Fare')
fig.show()
missingno.matrix(new_train_data)
missingno.matrix(test_data)
new_train_data['Cabin'].value_counts()
test_data['Cabin'].value_counts()
new_train_data_cabin = new_train_data.copy()
new_test_data = test_data.copy()
fare_mean = new_test_data['Fare'].mean()
new_test_data['Fare'] = new_test_data['Fare'].replace(np.NAN,fare_mean)
missingno.matrix(new_test_data)
new_train_data_cabin['Cabin_type'] = new_train_data_cabin['Cabin'].str.split(r'[0-9]').str[0].tolist()
new_test_data['Cabin_type'] = new_test_data['Cabin'].str.split(r'[0-9]').str[0].tolist()
new_train_data_cabin
new_test_data
new_train_data_cabin['Cabin_type'].value_counts()
new_test_data['Cabin_type'].value_counts()
lb1 = LabelEncoder()
new_train_data_cabin['Cabin_type'] = lb1.fit_transform(new_train_data_cabin['Cabin_type'].astype(str))
new_test_data['Cabin_type'] = lb1.fit_transform(new_test_data['Cabin_type'].astype(str))
new_train_data_cabin['Cabin_type'].value_counts()
new_train_data_cabin['Cabin_type'].value_counts()
new_train_data_cabin['Cabin_type'] = new_train_data_cabin['Cabin_type'].replace(0,11)
new_train_data_cabin['Cabin_type'] = new_train_data_cabin['Cabin_type'].replace(10,0)
new_train_data_cabin['Cabin_type'] = new_train_data_cabin['Cabin_type'].replace(11,10)
new_test_data['Cabin_type'] = new_test_data['Cabin_type'].replace(0,11)
new_test_data['Cabin_type'] = new_test_data['Cabin_type'].replace(10,0)
new_test_data['Cabin_type'] = new_test_data['Cabin_type'].replace(11,10)
new_train_data_cabin['Cabin_type'].value_counts()
new_test_data['Cabin_type'].value_counts()
new_train_data_cabin.head()
new_test_data.head()
new_train_data_cabin['Family'] = new_train_data_cabin['SibSp'] + new_train_data_cabin['Parch'] + 1
new_train_data_cabin
new_test_data['Family'] = new_test_data['SibSp'] + new_test_data['Parch'] + 1
new_test_data
new_test_data.drop('Name',axis=1,inplace=True)
new_test_data['Sex'] = pd.get_dummies(new_test_data['Sex'])
new_test_data
len(set(test_data.Ticket) - set(test_data.Ticket).intersection(set(new_train_data_cabin.Ticket))),len(set(test_data.Ticket)),len(test_data)
len(set(new_test_data.Cabin_type) - set(new_test_data.Cabin_type).intersection(set(new_train_data_cabin.Cabin_type))),len(set(new_test_data.Cabin_type)),len(new_test_data)
group = new_train_data_cabin.groupby(['Sex']).agg({'Fare':['mean']})
group.columns = ['mean_fare_sex']
group.reset_index(inplace=True)
new_train_data_cabin = pd.merge(new_train_data_cabin,group,on=['Sex'], how='left')
new_train_data_cabin
group = new_test_data.groupby(['Sex']).agg({'Fare':['mean']})
group.columns = ['mean_fare_sex']
group.reset_index(inplace=True)
new_test_data = pd.merge(new_test_data,group,on=['Sex'], how='left')
new_test_data
new_train_data_cabin.drop('Ticket',axis=1, inplace=True)
new_train_data_cabin
new_test_data.drop('Ticket',axis=1, inplace=True)
new_test_data
new_train_data_cabin.drop('PassengerId',axis=1,inplace=True)
new_test_data.drop('PassengerId',axis=1,inplace=True)
new_train_data_cabin['Pclass'].value_counts()
new_train_data_cabin.drop('Cabin',axis=1,inplace=True)
new_train_data_cabin
new_test_data.drop('Cabin',axis=1,inplace=True)
new_test_data
bins = np.linspace(min(new_train_data_cabin['Age']),max(new_train_data_cabin['Age']),4)
group_names = [1,2,3]
new_train_data_cabin['Age_binned'] = pd.cut(new_train_data_cabin['Age'],bins,labels=group_names,include_lowest=True)
new_train_data_cabin
bins = np.linspace(min(new_test_data['Age']),max(new_test_data['Age']),4)
group_names = [1,2,3]
new_test_data['Age_binned'] = pd.cut(new_test_data['Age'],bins,labels=group_names,include_lowest=True)
new_test_data
bins = np.linspace(min(new_train_data_cabin['Fare']),max(new_train_data_cabin['Fare']),4)
group_names = [1,2,3]
new_train_data_cabin['Fare_binned'] = pd.cut(new_train_data_cabin['Fare'],bins,labels=group_names,include_lowest=True)
new_train_data_cabin
bins = np.linspace(min(new_test_data['Fare']),max(new_test_data['Fare']),4)
group_names = [1,2,3]
new_test_data['Fare_binned'] = pd.cut(new_test_data['Fare'],bins,labels=group_names,include_lowest=True)
new_test_data
new_train_data_cabin.duplicated().sum()
new_train_data_cabin.drop_duplicates(inplace=True)
new_train_data_cabin.duplicated().sum()
new_train_data_cabin
training = new_train_data_cabin.copy()
training
training
training.drop('C',axis=1,inplace=True)
training.drop('Q',axis=1,inplace=True)
training.drop('S',axis=1,inplace=True)
training
x_data = training.drop('Survived',axis=1,inplace=False) #Independent Feature
y_data = training[['Survived']] #Dependent Feature
x_data.head()
y_data.head()
x_data.drop('mean_fare_sex',axis=1,inplace = True)
x_data['Age_binned']= x_data['Age_binned'].astype('int16')
x_data['Fare_binned']= x_data['Fare_binned'].astype('int16')
x_data.dtypes
new_test_data.drop('mean_fare_sex',axis=1,inplace = True)
new_test_data['Age_binned']= new_test_data['Age_binned'].astype('int16')
new_test_data['Fare_binned']= new_test_data['Fare_binned'].astype('int16')
new_test_data.dtypes
x_data.describe()
############################## Decision Tree Classifier ##################################
Dt = DecisionTreeClassifier()
Dt.fit(x_data,y_data)
y_pred = Dt.predict(x_data)
accuracy_score(y_data,y_pred)*100
## Feature Importance
feature = []
importance = []
for i, column in enumerate(x_data):
    print("The Feature importance for {} is : {}".format(column,Dt.feature_importances_[i]))
    feature.append(column)
    importance.append(Dt.feature_importances_[i])
plt.figure(figsize=(8,8))
sns.barplot(x=importance,y=feature)
plt.show()
rto =  RandomForestClassifier(n_estimators=28)
rto.fit(x_data,y_data.values.ravel())
y_pred = rto.predict(x_data)
accuracy_score(y_data,y_pred)*100
## Feature Importance
feature = []
importance = []
for i, column in enumerate(x_data):
    print("The Feature importance for {} is : {}".format(column,rto.feature_importances_[i]))
    feature.append(column)
    importance.append(rto.feature_importances_[i])
plt.figure(figsize=(8,8))
sns.barplot(x=importance,y=feature)
plt.show()
new_test_data
new_test_data['Embarked'] = lb1.fit_transform(new_test_data['Embarked'])
new_test_data
x_data.dtypes
x_data1 = x_data.drop('Fare_binned',axis= 1,inplace= False)
x_data1
x_data1.drop('Parch',axis=1,inplace=True)
x_data1
new_test_data1 = new_test_data.drop('Fare_binned',axis= 1,inplace= False)
new_test_data1
new_test_data1.drop('Parch',axis=1,inplace=True)
new_test_data1
model_names = {
    'svm' : {
     'model': SVC(gamma='auto'),
     'params': {
         'C': [1,5,10,26,28,29,30],
         'kernel' : ['rbf','linear']
     }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params' : {
            'C': [1,5,10,26,28,29,30],
        }
    },
    'random_forest' : {
        'model' : RandomForestClassifier(),
        'params' : {
            'n_estimators' : [1,5,10,26,28,29,30]
        }
    },
    'xgboost' : {
        'model' : xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5),
        'params' : {
            'alpha' : [0.1,1,10,100,1.1],
            'n_estimators' : [1,5,10,26,28,29,30]
        }
        
    },
    'bagging_classifier' : {
        'model' : BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5,max_features=1.0),
        'params' : {
            'n_estimators' : [1,5,10,26,28,29,30]
        }
    },
    'adbboost_classifier' : {
        'model' :  AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=1),
        'params' : {
            'n_estimators' : [1,5,10,26,28,29,30]
        }
    }
}
from sklearn.model_selection import GridSearchCV
scores = []
for model_name, mp in model_names.items():
    clf = GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=False)
    clf.fit(x_data1,y_data.values.ravel())
    scores.append({
        'model' : model_name,
        'best_score' : clf.best_score_,
        'best_params' : clf.best_params_
    })
d=pd.DataFrame(scores,columns=['model','best_score','best_params'])
d
px.bar(data_frame=d,x='model',y='best_score')
estimator3 = [] 
estimator3.append(('svm', SVC(gamma='auto',kernel='linear',C=1)))
estimator3.append(('lg', LogisticRegression(solver='liblinear',multi_class='auto',C=1)))
estimator3.append(('rfc', RandomForestClassifier(n_estimators=28))) 
estimator3.append(('xgb', xgb.XGBClassifier(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha=1.1,n_estimators=28))) 
estimator3.append(('bg', BaggingClassifier(DecisionTreeClassifier(), max_samples= 0.5,max_features=1.0,n_estimators=28))) 
estimator3.append(('ada2', AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=5,learning_rate=1))) 

votin = VotingClassifier(estimators=estimator3,voting='hard')
votin.fit(x_data1,y_data.values.ravel())
predic = votin.predict(new_test_data1)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predic})
output.to_csv('my_submission22.csv', index=False)
print("Your submission was successfully saved!")

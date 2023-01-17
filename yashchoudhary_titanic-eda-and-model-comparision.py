import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

import tensorflow as tf 

%matplotlib inline

import warnings;

warnings.filterwarnings("ignore");

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/train.csv")

test_sur = pd.read_csv("../input/gender_submission.csv")

test = pd.read_csv("../input/test.csv")
data.head()
data.info()
data.isna().sum()
import missingno as msno

msno.bar(data.sample(890))

msno.matrix(data)
data = data.drop(columns = ['Name','Ticket','Cabin','PassengerId'])

data['Age'][np.isnan(data['Age'])] = np.nanmedian(data['Age'])

data = data.dropna()
test_sur.head()
test.head()
test = test.drop(columns = ['Name','Ticket','Cabin','PassengerId'])

test['Age'][np.isnan(test['Age'])] = np.nanmedian(test['Age'])

test['Fare'][np.isnan(test['Fare'])] = np.nanmedian(test['Fare'])

test.head()
data.head()
test_df = data.copy()

map1 = {"female":0 , "male":1}

map2 = {"S":0 , "C":1 , "Q":2}

test_df['Sex']=test_df.Sex.map(map1)

test_df["Embarked"] = test_df.Embarked.map(map2)

corr_map = test_df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(corr_map,vmax=.7, square=True,annot=True,fmt=".2f",cmap='Blues')
sns.scatterplot(data["Fare"],data["Age"],color='Green')
sns.catplot(x="Survived", y="Age", hue="Sex", kind="swarm", data=data, aspect=1,height=8);
sns.factorplot(x="Sex",col="Survived", data=data , kind="count",size=7, aspect=.7,palette=['red','green'])
sns.catplot(x="Survived",hue="Pclass", kind="count",col='Sex', data=data,color='Violet',aspect=0.7,height=7);
sns.catplot(x="Survived", hue="SibSp", col = 'Sex',kind="count", data=data,height=7);

sns.catplot(x="Survived", hue="Parch", col = 'Sex', kind="count", data=data,height=7);
emb =data.groupby('Embarked').size()



plt.pie(emb.values,labels = ["Cherbourg","Queenstown","Southampton"],startangle=90,autopct='%1.1f%%');
pd.crosstab([data.Sex,data.Survived],data.Pclass, margins=True).style.background_gradient(cmap='gist_rainbow')
data.describe()
surv =data.groupby('Survived').size()

emb_sur = data[data['Survived']==1].groupby('Embarked').size()

emb_die = data[data['Survived']==0].groupby('Embarked').size()



pie_index = ['Cherbourg','Queenstown','Southampton']



fig = plt.figure()



ax1 = fig.add_axes([0, 0, 1, 1], aspect=1)

ax1.pie(surv.values,labels=['Died','Survived'],startangle=90,autopct='%1.1f%%')

plt.title('Survivors to Casualties Ratio',bbox={'facecolor':'0.8', 'pad':5})



ax2 = fig.add_axes([1, 0, 1, 1], aspect=1)

plt.title('Survivors percentage',bbox={'facecolor':'0.8', 'pad':5})

ax2.pie(emb_sur.values,labels = pie_index,startangle = 90, autopct='%1.1f%%')



ax3 = fig.add_axes([2, 0, 1, 1], aspect=1)

plt.title('Casualties percentage',bbox={'facecolor':'0.8', 'pad':5})

ax3.pie(emb_die.values,labels = pie_index,startangle = 90, autopct='%1.1f%%')



plt.show()      
sns.catplot(x="Embarked",hue="Survived", kind="count",col='Sex', data=data,aspect=0.7,height=7);
emb_smt = data[data['Embarked']=='S'].groupby('Survived').size()

emb_que = data[data['Embarked']=='Q'].groupby('Survived').size()

emb_che = data[data['Embarked']=='C'].groupby('Survived').size()



fig = plt.figure()

ax1 = fig.add_axes([0, 0, 1,1], aspect=1)

ax1.pie(emb_smt.values,labels = ['Died','Survived'],startangle = 90, autopct='%1.1f%%')

plt.title('Southampton',bbox={'facecolor':'0.8', 'pad':5})

ax2 = fig.add_axes([1, 0, 1,1], aspect=1)

ax2.pie(emb_que.values,labels = ['Died','Survived'],startangle = 90, autopct='%1.1f%%')

plt.title('Queenstown',bbox={'facecolor':'0.8', 'pad':5})

ax3 = fig.add_axes([2,0, 1,1], aspect=1)

ax3.pie(emb_che.values,labels = ['Died','Survived'],startangle = 90, autopct='%1.1f%%')

plt.title('Cherbourg',bbox={'facecolor':'0.8', 'pad':5})

plt.show()      
#Converting text data to numerical

map1 = {"female":0 , "male":1}

map2 = {"S":0 , "C":1 , "Q":2}

data['Sex']=data.Sex.map(map1)

data["Embarked"] = data.Embarked.map(map2)

test['Sex']=test.Sex.map(map1)

test["Embarked"] = test.Embarked.map(map2)

#one hot encoding

data = pd.get_dummies(data)

test = pd.get_dummies(test)
data.head()
test.head()
data['Family'] = data.Parch+data.SibSp

data.Age = data.Age/np.mean(data.Age)

data.Fare = data.Fare/np.mean(data.Fare)

test['Family'] = test.Parch+test.SibSp

test.Age = test.Age/np.mean(test.Age)

test.Fare = test.Fare/np.mean(test.Fare)
data.head()
test.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,1:],data.iloc[:,0],test_size=0.2)

#using 70:30 split.
print(x_train.head())

print(y_train.head())
classifiers=[['Logistic Regression :',LogisticRegressionCV()],

             ['SVM:',svm.LinearSVC()],

       ['Decision Tree Classification :',DecisionTreeClassifier()],

       ['Random Forest Classification :',RandomForestClassifier()],

       ['Gradient Boosting Classification :', GradientBoostingClassifier()],

       ['Ada Boosting Classification :',AdaBoostClassifier()],

       ['Extra Tree Classification :', ExtraTreesClassifier()],

       ['K-Neighbors Classification :',KNeighborsClassifier()],

       ['Support Vector Classification :',SVC()],

       ['Gaussian Naive Bayes :',GaussianNB()]]

cla_pred=[]

for name,model in classifiers:

    model=model

    model.fit(x_train,y_train)

    predictions = model.predict(x_test)

    cla_pred.append(accuracy_score(y_test,predictions))

    print(name,accuracy_score(y_test,predictions))
plt.bar(x=[1,2,3,4,5,6,7,8,9,10], height=np.multiply(cla_pred,100)

        ,tick_label=['LR','SVM','DTC', 'RFC', 'GBC', 'ABC', 'ETC', 'KNN', 'SVC','GNB']

        , color=["Blue","Green","red","orange","Yellow","cyan","pink","purple","black","violet"])

plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

gen_sub = pd.read_csv("../input/gender_submission.csv")
train.head()
test.head()
gen_sub.head()
test.info()
#removing non useful columns

#train

train_clean = train.iloc[:,[1,2,4,5,6,7,9,11]]

# train_clean['Sex'].replace('male', 0,inplace=True)

# train_clean['Sex'].replace('female', 1,inplace=True)

train_clean['Sex'] = train_clean['Sex'].astype('category')

train_clean['Sex'] = train_clean['Sex'].cat.codes

# train_clean['Embarked'].replace('S', 0,inplace=True)

# train_clean['Embarked'].replace('C', 1,inplace=True)

# train_clean['Embarked'].replace('Q', 2,inplace=True)

train_clean.reset_index(inplace=True, drop = True)

train_clean = train_clean[train_clean['Embarked'].notna()]

categorical = ['Embarked']



for var in categorical:

    train_clean = pd.concat([train_clean, 

                    pd.get_dummies(train_clean[var], prefix=var)], axis=1)

    del train_clean[var]

train_clean.reset_index(inplace=True, drop = True)

train_clean['Age'].fillna(train_clean['Age'].mean(), inplace=True)

train_clean.reset_index(inplace=True, drop = True)

train_clean['Family_Size']= train_clean['SibSp']+train_clean['Parch']









#test

test_clean = test.iloc[:,[0,1,3,4,5,6,8,10]]

# test_clean['Sex'].replace('male', 0,inplace=True)

# test_clean['Sex'].replace('female', 1,inplace=True)

test_clean['Sex'] = test_clean['Sex'].astype('category')

test_clean['Sex'] = test_clean['Sex'].cat.codes

# test_clean['Embarked'].replace('S', 0,inplace=True)

# test_clean['Embarked'].replace('C', 1,inplace=True)

# test_clean['Embarked'].replace('Q', 2,inplace=True)

test_clean.reset_index(inplace=True, drop = True)

test_clean = test_clean[test_clean['Embarked'].notna()]

categorical = ['Embarked']



for var in categorical:

    test_clean = pd.concat([test_clean, 

                    pd.get_dummies(test_clean[var], prefix=var)], axis=1)

    del test_clean[var]

    

test_clean.reset_index(inplace=True, drop = True)

test_clean['Age'].fillna(test_clean['Age'].mean(), inplace=True)

test_clean['Fare'].fillna(test_clean['Fare'].mean(), inplace=True)

test_clean.reset_index(inplace=True, drop = True)

test_clean['Family_Size']= test_clean['SibSp']+test_clean['Parch']
train_clean.head()
train_clean.info()
test_clean.head()
test_clean.info()
# #Let's normalize these dataframe



# x = train_clean.values #returns a numpy array

# min_max_scaler = preprocessing.MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# train_clean = pd.DataFrame(x_scaled,columns=train_clean.columns)



# x = test_clean.values #returns a numpy array

# min_max_scaler = preprocessing.MinMaxScaler()

# x_scaled = min_max_scaler.fit_transform(x)

# test_clean = pd.DataFrame(x_scaled,columns=test_clean.columns)

from sklearn.preprocessing import StandardScaler



continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Family_Size']



scaler = StandardScaler()



for var in continuous:

    train_clean[var] = train_clean[var].astype('float64')

    train_clean[var] = scaler.fit_transform(train_clean[var].values.reshape(-1, 1))



for var in continuous:

    test_clean[var] = test_clean[var].astype('float64')

    test_clean[var] = scaler.fit_transform(test_clean[var].values.reshape(-1, 1))
train_clean.head()
test_clean.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_clean.iloc[:,1:],train_clean.iloc[:,0],test_size=0.2)
import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout



def clf_model():

    model=Sequential()

    model.add(Dense(10,input_dim=10,kernel_initializer='normal',activation='relu'))

    model.add(Dense(16,activation='relu'))

    model.add(Dense(32,activation='relu'))

    model.add(Dense(16,activation='relu'))

    model.add(Dense(1,activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



model = clf_model()
model.summary()
from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

filepath1="best_model.hdf5"

checkpoint = ModelCheckpoint(filepath1, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(

    monitor='val_loss',

    patience=4,

    verbose=1,

    min_lr=1e-6

)

callbacks_list = [checkpoint,reduce_lr]
model.fit(x_train,y_train,validation_data = (x_test,y_test),epochs=100, batch_size=16, callbacks=callbacks_list, verbose=1)
model.load_weights('best_model.hdf5') #select the best weights file
gen_sub['Survived'] = model.predict_classes(test_clean.iloc[:,1:])
gen_sub.info()
gen_sub.head(10)
gen_sub.to_csv("submit.csv", index=False)
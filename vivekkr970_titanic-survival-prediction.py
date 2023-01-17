import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.tree import export_graphviz,DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

%matplotlib inline
mypath="../input/"
train_data=pd.read_csv(mypath+'train.csv')

test_data=pd.read_csv(mypath+'test.csv')

test_survived_data=pd.read_csv(mypath+'gender_submission.csv')

train_data.head()
#compute basic statistic on data

train_data.describe()
print(train_data.info())

print(test_data.info())
#lets pplot a histogram to have a previsualization of some of data

train_data.drop(['PassengerId'],1).hist(bins=50, figsize=(20,15))

plt.show()
#with this first exploration of training data we can see that:

#1. only approx 35% of pessenger survived

#2. more than half of pessenger are in lowest Pclass =3

#3. most of fare ticket are below 50

#4. majority of pessenger are alone (sibsp and parch)
test_data.drop(['PassengerId'],1).hist(bins=50, figsize=(20,15))

plt.show()
#there is 77% of data missing in cabin column, it's way to much for column to be exploitable, 

#but as we have small amount of data we will still use it is feature engineering

#there is only two missing value for embarked column, lets try to fill it

# below is distribution of embarked according to fare and sex

from IPython.display import display

plot=sns.catplot(x='Embarked',y='Fare',hue='Sex',data=train_data,kind='bar')

display(train_data[train_data['Embarked'].isnull()])

#train_data['Embarked'].fillna(str(train_data['Embarked'].mode().values[0]),inplace=True)
#both pessenger are female who paid 80$ as fare for tickets, also they have same ticket and cabin, 

#so they probably had to board at same place, according to ablove distribution the more probable embaeked value for them is c

train_data['Embarked'].fillna('C',inplace=True)
# we replace missing age by mean age of pessenger who belong to some group of class/sex/family

train_data.Age=train_data.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))

train_data.Age=train_data.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

train_data.Age=train_data.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))



train_data.tail(10)
test_data.Age=test_data.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))

test_data.Age=test_data.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

test_data.Age=test_data.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))



test_data.head()
# we replace the missing value of fare for test dataset using interpolation

display(test_data[test_data['Fare'].isnull()])
test_data.Fare=test_data['Fare'].interpolate()
test_data['Cabin'].fillna('U',inplace=True)

train_data['Cabin'].fillna('U',inplace=True)
print(train_data.info())

print(test_data.info())
plt.figure(figsize = (16,5))

sns.heatmap(train_data.corr(),cmap="BrBG",annot=True)
#create a title column from name column

train_data['Title']=pd.Series((name.split('.')[0].split(',')[1].strip() for name in train_data['Name']), index=train_data.index)

print(train_data['Title'].unique())



test_data['Title']=pd.Series((name.split('.')[0].split(',')[1].strip() for name in test_data['Name']), index=test_data.index)

print(test_data['Title'].unique())
train_data['Title']=train_data['Title'].replace(['Don', 'Rev', 'Dr','Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess',

       'Jonkheer'],'Rare')

train_data['Title']=train_data['Title'].replace(['Mlle','Ms'],'Miss')

train_data['Title']=train_data['Title'].replace('Mme','Mrs')

train_data['Title']=train_data['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})



test_data['Title']=test_data['Title'].replace(['Dona', 'Rev', 'Dr', 'Col'],'Rare')

test_data['Title']=test_data['Title'].replace(['Mlle','Ms'],'Miss')

test_data['Title']=test_data['Title'].replace('Mme','Mrs')

test_data['Title']=test_data['Title'].map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})
#filling age missing value with mean age of passengers who have same title

train_data['Age']=train_data.groupby(['Title'])['Age'].transform(lambda x:x.fillna(x.mean()))

test_data['Age']=test_data.groupby(['Title'])['Age'].transform(lambda x:x.fillna(x.mean()))
#transform categorical variable to numerical variable

train_data['Sex']=train_data['Sex'].map({'male':0,'female':1})

train_data['Embarked']=train_data['Embarked'].map({'S':0,'C':1,'Q':2})



test_data['Sex']=test_data['Sex'].map({'male':0,'female':1})

test_data['Embarked']=test_data['Embarked'].map({'S':0,'C':1,'Q':2})
#modification of cabin column to keep only letter contained corresponding to  deck of boat

train_data['Cabin']=train_data['Cabin'].str[:1]

train_data['Cabin']=train_data['Cabin'].map({cabin: p for p,cabin in enumerate(set(cab for cab in train_data['Cabin']))})



test_data['Cabin']=test_data['Cabin'].str[:1]

test_data['Cabin']=test_data['Cabin'].map({cabin: p for p,cabin in enumerate(set(cab for cab in test_data['Cabin']))})
train_data.head()
#create a family size. isalone, child and mother column

train_data['FamilySize']=train_data['SibSp']+train_data['Parch']+1

train_data['FamilySize'][train_data['FamilySize'].between(1,5,inclusive=False)]=2

train_data['FamilySize'][train_data['FamilySize']>5]=3

train_data['IsAlone']=np.where(train_data['FamilySize']!=1,0,1)



test_data['FamilySize']=test_data['SibSp']+test_data['Parch']+1

test_data['FamilySize'][test_data['FamilySize'].between(1,5,inclusive=False)]=2

test_data['FamilySize'][test_data['FamilySize']>5]=3

test_data['IsAlone']=np.where(test_data['FamilySize']!=1,0,1)
def persontype_func_num(age_gender):

    age, gender = age_gender

    if age < 16:

        m=2

    else:

        if gender == 1:

            m=1

        else: 

            m=0

    return m

train_data['PersonType_num'] = train_data[['Age', 'Sex']].apply(persontype_func_num, axis=1)

test_data['PersonType_num'] = test_data[['Age', 'Sex']].apply(persontype_func_num, axis=1)
# in case of Pclass and FamilySize their order has some meaning so we will not create dummies for them

Embarked_dummies=pd.get_dummies(train_data['Embarked'],prefix='Embarked')

PersonType_num_dummies=pd.get_dummies(train_data['PersonType_num'],prefix='PersonType_num')

Title_dummies=pd.get_dummies(train_data['Title'],prefix='Title')

Cabin_dummies=pd.get_dummies(train_data['Cabin'],prefix='Cabin')

train_data = pd.concat([train_data,Embarked_dummies,PersonType_num_dummies,Title_dummies,Cabin_dummies],axis=1)

#drop first dummy column as it will take care by other dummy columns

train_data.drop('Embarked_0',axis=1)

train_data.drop('PersonType_num_0',axis=1)

train_data.drop('Title_0',axis=1)

train_data.drop('Cabin_0',axis=1)

train_data.head()
# in case of Pclass and FamilySize their order has some meaning so we will not create dummies for them

Embarked_dummies=pd.get_dummies(test_data['Embarked'],prefix='Embarked')

PersonType_num_dummies=pd.get_dummies(test_data['PersonType_num'],prefix='PersonType_num')

Title_dummies=pd.get_dummies(test_data['Title'],prefix='Title')

Cabin_dummies=pd.get_dummies(test_data['Cabin'],prefix='Cabin')

test_data = pd.concat([test_data,Embarked_dummies,PersonType_num_dummies,Title_dummies,Cabin_dummies],axis=1)

#drop first dummy column as it will take care by other dummy columns

test_data.drop('Embarked_0',axis=1)

test_data.drop('PersonType_num_0',axis=1)

test_data.drop('Title_0',axis=1)

test_data.drop('Cabin_0',axis=1)

test_data.head()
# environment settings: 

pd.set_option('display.max_column',None)

pd.set_option('display.max_rows',None)

pd.set_option('display.max_seq_items',None)

pd.set_option('display.max_colwidth', 500)

pd.set_option('expand_frame_repr', True)

print(train_data.columns)

print(test_data.columns)
# create a list of features column

feature_cols = ['Sex', 'Age', 'SibSp','Parch', 'Pclass', 'FamilySize','IsAlone', 'Embarked_1', 'Embarked_2', 'PersonType_num_1',

       'PersonType_num_2','Title_1', 'Title_2', 'Title_3','Title_4', 'Cabin_1', 'Cabin_2', 'Cabin_3', 'Cabin_4',

       'Cabin_5', 'Cabin_6', 'Cabin_7']

# create x and y for training and test data

x = train_data[feature_cols]

y = train_data['Survived']



x_test=test_data[feature_cols]

y_test=test_survived_data.Survived
#set benchmark for rmse results for all models compared to using statistical mean

# create a NumPy array with the same shape as y_test

y_null = np.zeros_like(y_test, dtype=float)



# fill the array with the mean value of y_test

y_null.fill(y_test.mean())

# compute null RMSE

np.sqrt(metrics.mean_squared_error(y_test, y_null))
#using simple logistic regression



#create and train model on train data sample

lg=LogisticRegression(random_state=42)

lg.fit(x,y)



#predict for test data sample

logistion_prediction=lg.predict(x_test)



# compute RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, logistion_prediction)))



# compute error between predicted data and true responce and display it in confusion matrix

print(metrics.accuracy_score(y_test,logistion_prediction))

print(metrics.classification_report(y_test,logistion_prediction))

print(confusion_matrix(y_test,logistion_prediction))

sns.heatmap(confusion_matrix(y_test,logistion_prediction),cmap="BrBG",annot=True)
#using Decision Tree



#create and train model on train data sample

dt=DecisionTreeClassifier(min_samples_split=15,min_samples_leaf=20,random_state=42)

dt.fit(x,y)



#predict for test data sample

decision_prediction=dt.predict(x_test)



# compute RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, decision_prediction)))



# compute error between predicted data and true responce and display it in confusion matrix

print(metrics.accuracy_score(y_test,decision_prediction))

print(metrics.classification_report(y_test,decision_prediction))

print(confusion_matrix(y_test,decision_prediction))

sns.heatmap(confusion_matrix(y_test,decision_prediction),cmap="BrBG",annot=True)
#using Random Forest



#create and train model on train data sample

rf=RandomForestClassifier(n_estimators=200,random_state=42)

rf.fit(x,y)



#predict for test data sample

rf_prediction=rf.predict(x_test)



# compute RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, rf_prediction)))



# compute error between predicted data and true responce and display it in confusion matrix

print(metrics.accuracy_score(y_test,rf_prediction))

print(metrics.classification_report(y_test,rf_prediction))

print(confusion_matrix(y_test,rf_prediction))

sns.heatmap(confusion_matrix(y_test,rf_prediction),cmap="BrBG",annot=True)
#using SVM



#create and train model on train data sample

svm=SVC(gamma='auto',random_state=42)

svm.fit(x,y)



#predict for test data sample

svm_prediction=svm.predict(x_test)



# compute RMSE

print(np.sqrt(metrics.mean_squared_error(y_test, svm_prediction)))



# compute error between predicted data and true responce and display it in confusion matrix

print(metrics.accuracy_score(y_test,svm_prediction))

print(metrics.classification_report(y_test,svm_prediction))

print(confusion_matrix(y_test,svm_prediction))

sns.heatmap(confusion_matrix(y_test,svm_prediction),cmap="BrBG",annot=True,fmt='g')
test_survived_data['SVM_Prediction']=svm_prediction

test_survived_data.to_csv('TitanicResult.csv',index=False)
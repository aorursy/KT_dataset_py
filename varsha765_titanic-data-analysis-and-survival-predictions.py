#import useful libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sb
#load dataset
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.describe(include='all')
#by looking at the data we conclude that age has some missing data but it seems to be an important feature to determine survival chances so we need to add the missing data.
#cabin data has a lot of missing values so it will be better to simply remove it from our feature set.
#Embarked data has only very few data points missing so that can be easily worked out.
#analyzing data type for features
train.info()
#name, sex, tickets, cabin(to be neglected as explained above), embarked features have string type data while rest all the features have numeric data
#handling missing data
train.isna().values.any()
#true shows there are missing data points in our data set
train.isna().sum()
test.isna().sum()
#age, fare, cabin has missing values is test dataset
#visualizing missing data 
missing_data=(train.isna().sum()/len(train))*100   #percent missing data
sb.barplot(x=missing_data.index, y=missing_data)
plt.xlabel('Features')
plt.ylabel('% missing data')
plt.xticks(rotation=90)
plt.show()

#replacing missing data in age 
mean=train['Age'].mean()
train['Age'].fillna(mean, inplace=True)
#train['Age'].isna().values.any()    #should return False
mean_test=test['Age'].mean()
test['Age'].fillna(mean_test, inplace=True)
#test['Age'].isna().sum()     #should return 0
train['Cabin'].fillna('Unknown', inplace=True)
train['Embarked'].fillna('Unknown', inplace=True)
train.isna().sum()     
#no missing values anymore in train

test['Cabin'].fillna('Unknown', inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test.isna().sum()
#analyzing and comparing the survived and drowned fractions from the train data
status=train['Survived'].value_counts()
status.index=['Drowned', 'Survived']
sb.barplot(status.index, status)
plt.xlabel('status of passangers')
plt.ylabel('number of passangers')
plt.show()

#analyzing which sex had higher chances of survival
#sb.barplot(x=train['Sex'], y=train['Survived'])
#plt.show()
gen=train.groupby(['Survived'])['Sex'].value_counts()
gen
#gender distribution for drowned people
sb.barplot(x=gen.index,y=gen)
plt.show()

#which class had more surviving chances
plt.subplots()
sb.countplot('Pclass',hue='Survived', data=train)
plt.show()
#analyzing survival based on age
#different age distribution
sb.distplot(train['Age'])
plt.ylabel('Fraction')
plt.show()
#we can see most of the people were between 20 amd 40 age group. 
#age based survival chances


bins = [0, 2, 10, 18, 35, 65, np.inf]
names = ['<2','2-10', '10-18', '18-35', '35-65', '65+']


x = pd.cut(train['Age'], bins, labels=names)
sb.countplot(x, hue='Survived', data=train)
plt.show()
#maximum 18-35 age group survived and drowned
#analyzing families (siblings, spouses, parents, children)
family_size=train['SibSp']+train['Parch']
sb.countplot(x,hue='Survived', data=train)
plt.show()
#most of the people were travelling alone. 
#People with 1, 2 or 3 family members had more chances of survival.

#feature engineering

#dealing with categorical data
train.info()
print(train['Sex'].unique())
print(train['Embarked'].unique())
#categorical data are Sex, Ticket, Embarked

total_data=train.append(test)
total_data.head(1)
#using labelEncoder to deal with categorical data
from sklearn.preprocessing import LabelEncoder
def transform(x):
    le=LabelEncoder()
    labels=le.fit_transform(x)
    mapping={index: label for index, label in enumerate(le.classes_)}
    print(mapping)
    return labels
total_data['Sex']=transform(total_data['Sex'])
total_data['Embarked']=transform(total_data['Embarked'])
total_data.head()
#we see sex and embarked now have numerical values.
#added new column family size.
total_data['Family_size']=total_data['SibSp']+total_data['Parch']+1
total_data.head(1)
#changing the cabin coloumn values to category of cabins
total_data['Cabin']=total_data['Cabin'].astype(str).str[0]
total_data['Cabin'].unique()
#converting cabin categories to numerical values
total_data['Cabin']=transform(total_data['Cabin'])

#dataset is preparesd with all necessary and required feature engineering
#now the task is to select which features to use in our model
#correlation will show how much features are interdependent.
sb.heatmap(total_data.corr(),annot=True,annot_kws={'size':8})
plt.show()
#we see family_size, SibSp, Parch are closely interdependent which should be obvious
#Cabin and pclass are also strongly interdependent.
#feature seclection

features=['Pclass','Sex','Age','Fare','Cabin','Embarked', 'Family_size']
train=total_data[: 891]
test=total_data[891: ]
train.head()
#total_data.isna().sum()
#building machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
df_train=train
df_test=test
X=df_train[features]
y=df_train['Survived'].astype(int)

#dividing training data into test set and training set to check accuracy of the implemented model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=324)
#I will be implementing two machine learning model to make the predictions and compare which one is giving more 
#more accurate predictions
#model_1: Random forest classifier
#model_2: Logistic regression
#implementing random forest machine learning model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
#implementing logistic regression
logreg = LogisticRegression(max_iter=400)
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)
#comparing accuracy
models=['Random forest','Logistic regression']
accuracy=[acc_random_forest, acc_log]
results=pd.DataFrame({'Models':models, 'Accuracy':accuracy})
results
randomForestFinalModel = RandomForestClassifier(random_state = 2, bootstrap=False,min_samples_split=2,min_samples_leaf= 5, criterion = 'entropy', max_depth = 13, max_features = 'sqrt', n_estimators = 200)
randomForestFinalModel.fit(X_train, y_train)
predictions_rf = randomForestFinalModel.predict(X_test)
#prediction values from random forest classifier
submission=pd.DataFrame({"PassengerId": df_test["PassengerId"],
    "Survived": randomForestFinalModel.predict( df_test[features])})
submission.head()
#prediction values from logistic regression 
s=logreg.predict(df_test[features])
s=pd.DataFrame({'PassengerID':df_test['PassengerId'], 'Survived': s})
s.head()
submission.to_csv('submission.csv',index=False)

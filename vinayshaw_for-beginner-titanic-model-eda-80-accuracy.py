# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

%matplotlib inline                 



import pandas as pd                # Implemennts milti-dimensional array and matrices

import numpy as np                 # For data manipulation and analysis

import matplotlib.pyplot as plt    # Plotting library for Python programming language and it's numerical mathematics extension NumPy

import seaborn as sns              # Provides a high level interface for drawing attractive and informative statistical graphics



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load dataset

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission=pd.read_csv("../input/titanic/gender_submission.csv")
len(train),len(test),len(gender_submission)
train.head()
train.shape
train.describe(include='all')
train.info()
train.isnull().values.any()
train.isnull().sum()
test.isnull().sum()
plt.style.use('default')

total=train.isnull().sum()

percent=train.isnull().sum()/train.isnull().count()

missing_data=pd.concat([total,percent],axis=1, keys=['total', 'percent'])

#missing_data.sort_values(ascending=False)

ax = plt.subplots(figsize=(12, 6))

#plt.xticks(rotation='90')

sns.barplot(x=missing_data.index,y=missing_data['percent'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)

plt.show()
import missingno as msno
msno.bar(train,figsize=(10,6),color="skyblue")

plt.show()
msno.bar(test,figsize=(10,6),color="skyblue")

plt.show()
msno.heatmap(train,figsize=(10,6))

plt.show()
msno.matrix(train,figsize=(12,8))

plt.show()
msno.matrix(test,figsize=(12,8))

plt.show()
train['Age'].fillna(train['Age'].median(),inplace=True)

test['Age'].fillna(train['Age'].median(),inplace=True)
train['Age']
train['Cabin'].unique()
train['Cabin'].fillna('Unknown',inplace=True)

train['Embarked'].fillna('Unknown',inplace=True)

test['Cabin'].fillna('Unknown',inplace=True)

test['Fare'].fillna(train['Fare'].median(),inplace=True)
msno.bar(train,figsize=(10,6),color="skyblue")

plt.show()
msno.bar(test,figsize=(10,6),color="skyblue")

plt.show()
msno.matrix(train,figsize=(12,6))

plt.show()
train['Survived'].value_counts(normalize=True)


sns.countplot(x='Survived',data=train)

plt.xticks( np.arange(2), ['drowned', 'survived'] )

plt.title('Overall survival (training dataset)',fontsize= 18)

# set x label

plt.xlabel('Passenger status after the tragedy',fontsize = 15)

# set y label

plt.ylabel('Number of passengers',fontsize = 15)

labels = (train['Survived'].value_counts())

for i, v in enumerate(labels):

    plt.text(i, v-40, str(v), horizontalalignment = 'center', size = 14, color = 'w', fontweight = 'bold')

    

plt.show()



sns.barplot(x = "Sex", y = "Survived", data=train)

plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize =10)

labels = ['Female', 'Male']

plt.ylabel("% of passenger survived", fontsize = 8)

plt.xlabel("Gender",fontsize = 8)

plt.show()

print("% of women survived: " , train[train.Sex == 'female'].Survived.sum()/train[train.Sex == 'female'].Survived.count())

print("% of men survived:   " , train[train.Sex == 'male'].Survived.sum()/train[train.Sex == 'male'].Survived.count())


sns.catplot(x='Sex', col='Survived', kind='count', data=train)



plt.show()
train.groupby(['Survived','Sex']).count()
train['Pclass'].unique()


plt.subplots(figsize = (8,6))

sns.countplot('Pclass',hue='Survived',data=train)



plt.show()
plt.subplots(figsize = (8,6))

sns.barplot('Pclass','Survived',data=train,hue='Sex',edgecolor=(0,0,0), linewidth=2)

plt.show()
sns.catplot('Pclass','Survived', kind='point', data=train);
plt.subplots(figsize=(8,6))

sns.kdeplot(train.loc[(train['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )



labels = ['First', 'Second', 'Third']

plt.xticks(sorted(train.Pclass.unique()),labels)

plt.show()
print("% of survivals in") 

print("Pclass=1 : ", train.Survived[train.Pclass == 1].sum()/train.Survived[train.Pclass == 1].count())

print("Pclass=2 : ", train.Survived[train.Pclass == 2].sum()/train.Survived[train.Pclass == 2].count())

print("Pclass=3 : ", train.Survived[train.Pclass == 3].sum()/train[train.Pclass == 3].Survived.count())
plt.subplots(figsize=(8,6))

sns.distplot(train.Age)

plt.title('Distrubution of passengers age (all data)',fontsize= 14)

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.show()
bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train,ci=None)

plt.show()
train.Name.head()
train['Title'] = train['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')

test['Title'] = test['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')

plt.figure(figsize=(8, 6))

ax = sns.countplot( x = 'Title', data = train, palette = "hls", order = train['Title'].value_counts().index)

_ = plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)



plt.title('Passengers distribution by titles',fontsize= 14)

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (train['Title'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+10, str(v), horizontalalignment = 'center', size = 10, color = 'black')



plt.show()
plt.figure(figsize=(10, 6))

sns.barplot(x="Title", y="Survived", data=train,ci=None) 

plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)



plt.show()
train['Cabin']
train['Cabin'].unique()

train['deck']=train['Cabin'].str.split('',expand=True)[1]

test['deck']=test['Cabin'].str.split('',expand=True)[1]
train['deck'].unique()
plt.figure(figsize=(12,8))

sns.countplot(x=train['deck'],data=train,hue='Survived',order = train['deck'].value_counts().index)

plt.title('Passengers distribution by deck',fontsize= 16)

plt.ylabel('Number of passengers')

plt.legend(( 'Drowned', 'Survived'), loc=(0.85,0.89))

plt.xticks(rotation = False)





plt.show()
#draw a bar plot for Parch vs. survival

plt.figure(figsize=(8,6))

sns.barplot(x="Parch", y="Survived", data=train,ci=None)

plt.show()
#draw a bar plot for SibSp vs. survival

sns.barplot(x="SibSp", y="Survived", data=train,ci=None)

plt.show()
train['SibSp'].sort_values().unique()
print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 3 who survived:", train["Survived"][train["SibSp"] == 3].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 4 who survived:", train["Survived"][train["SibSp"] == 4].value_counts(normalize = True)[1]*100)

plt.subplots(figsize=(8,6))



ax=sns.kdeplot(train.loc[(train['Survived'] == 0),'Fare'],color='r',shade=True,label='Not Survived')

ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'],color='b',shade=True,label='Survived' )

plt.title('Fare Distribution Survived vs Non Survived')

plt.ylabel('Frequency of Passenger Survived')

plt.xlabel('Fare')

plt.show()
sns.catplot(x="Pclass", y="Fare",hue='Survived', kind="swarm", data=train)

plt.show()
train['Embarked'].unique()
train['Embarked'].describe()
train['Embarked'] = train['Embarked'].replace('Unknown','S')
sns.countplot(train.Embarked)

labels = (train['Embarked'].value_counts())

plt.figure(figsize=(10,6))

sns.countplot(train['Embarked'],hue='Survived',data=train)

plt.legend(( 'Drowned', 'Survived'), loc=(0.85,0.89))

plt.show()
train.head()
total_data=train.append(test)
total_data.head()
total_data.shape
total_data['Sex'] =total_data['Sex'].replace('male',0)

total_data['Sex'] =total_data['Sex'].replace('female',1)

total_data['Embarked'] =total_data['Embarked'].replace('S',0)

total_data['Embarked'] = total_data['Embarked'].replace('Q',1)

total_data['Embarked'] = total_data['Embarked'].replace('C',2)
mapping = {'Mlle': 'Miss', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs',

           'Jonkheer': 'Rare', 'Lady': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Mrs', 'Rev':'Rare', 'Dr':'Rare'}



total_data.replace({'Title': mapping}, inplace=True)



total_data['Title'].value_counts(normalize=True)*100
total_data['Title'] = total_data['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2, 'Master':3, 'Rare':4})

total_data['Title'].fillna(total_data['Title'].median(),inplace=True)
cabin_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'T':8, 'U':9}

total_data['deck'] = total_data['deck'].map(cabin_category)

total_data['Family_size'] = total_data['SibSp'] + total_data['Parch'] + 1

total_data['Alone'] = 1

total_data['Alone'].loc[total_data['Family_size'] > 1] = 0
bins = [-1, 0, 18, 25, 35, 60, np.inf]

labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

total_data['AgeGroup'] = pd.cut(total_data["Age"], bins, labels = labels)

age_mapping = {'Unknown': None,'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}

total_data['AgeGroup'] = total_data['AgeGroup'].map(age_mapping)
fig,ax=plt.subplots(figsize=(14,6))

sns.heatmap(total_data.corr(),annot=True,annot_kws={'size':12})
total_data.head()
total_data.isna().sum()
features = ['Embarked','Fare','Pclass','Sex','Title','Family_size','Alone']
#Modelos

from sklearn.ensemble import RandomForestClassifier



#Metrics

from sklearn.metrics import make_scorer, accuracy_score,precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



#Model Select

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.linear_model import  LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
df_train = total_data[0:891]

df_test =  total_data[891:]

X = df_train[features]

y = df_train['Survived'].astype(int)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=78941)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)

logreg.fit(X_train, y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, y_train) * 100, 2)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test) 

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test) 

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, y_train)  

Y_pred = decision_tree.predict(X_test) 

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
results = pd.DataFrame({

    'Model': [ 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              ' Support Vector Machine', 

              'Decision Tree'],

    'Score': [ acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,  

              acc_linear_svc, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
model= LogisticRegression(solver= 'lbfgs',max_iter=400)

model.fit(X_train, y_train)

predictions = model.predict(X_test)





cm_logit = confusion_matrix(y_test, predictions)

print('Confusion matrix for Logistic\n',cm_logit)



accuracy_logit = accuracy_score(y_test,predictions)

precision_logit =precision_score(y_test, predictions)

recall_logit =  recall_score(y_test, predictions)

f1_logit = f1_score(y_test, predictions)

print('accuracy_logistic : %.3f' %accuracy_logit)

print('precision_logistic : %.3f' %precision_logit)

print('recall_logistic : %.3f' %recall_logit)

print('f1-score_logistic : %.3f' %f1_logit)

auc_logit = roc_auc_score(y_test,predictions)

print('AUC_logistic : %.2f' % auc_logit)
randomForestFinalModel = RandomForestClassifier(random_state = 2, bootstrap=False,min_samples_split=2,min_samples_leaf= 5, criterion = 'entropy', max_depth = 13, max_features = 'sqrt', n_estimators = 200)

randomForestFinalModel.fit(X_train, y_train)

predictions_rf = randomForestFinalModel.predict(X_test)



cm_logit = confusion_matrix(y_test, predictions_rf)

print('Confusion matrix for Random Forest\n',cm_logit)



accuracy_logit = accuracy_score(y_test,predictions_rf)

precision_logit =precision_score(y_test, predictions_rf)

recall_logit =  recall_score(y_test, predictions_rf)

f1_logit = f1_score(y_test,predictions_rf)

print('accuracy_random_Forest : %.3f' %accuracy_logit)

print('precision_random_Forest : %.3f' %precision_logit)

print('recall_random_Forest : %.3f' %recall_logit)

print('f1-score_random_Forest : %.3f' %f1_logit)

auc_logit = roc_auc_score(y_test,predictions_rf)

print('AUC_random_Forest: %.2f' % auc_logit)

a=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

b=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

fig =plt.figure(figsize=(20,12),dpi=50)

fpr, tpr, thresholds = roc_curve(y_test,predictions )

plt.plot(fpr, tpr,color ='orange',label ='Logistic',linewidth=2 )

fpr, tpr, thresholds = roc_curve(y_test,predictions_rf )

plt.plot(fpr, tpr,color ='blue',label ='random Forest',linewidth=2 )



plt.plot(a,b,color='black',linestyle ='dashed',linewidth=2)

plt.legend(fontsize=15)

plt.xlabel('False Positive Rate',fontsize=15)

plt.ylabel('True Positive Rate',fontsize=15)
submission = pd.DataFrame({

    "PassengerId": df_test["PassengerId"],

    "Survived": randomForestFinalModel.predict( df_test[features])

})

submission.head()
submission.to_csv("titanic_s.csv",index=False)
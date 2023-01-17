#importing all useful library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualizing
import seaborn as sns #powerful library for advance visualization.

from sklearn import preprocessing #for encoding categorical variable
from sklearn.linear_model import LogisticRegression #logistic classifier
from sklearn.svm import SVC #SVC classifier
from sklearn.ensemble import RandomForestClassifier #random forest classifier
from sklearn.neighbors import KNeighborsClassifier # k-nearst neighbour
from sklearn.naive_bayes import GaussianNB #GaussianNB classifier
from sklearn.tree import DecisionTreeClassifier #DecisionTree classifier

from sklearn.model_selection import train_test_split #for splitting data into training and testing.
from sklearn.metrics import f1_score #for checking f1-score of model
from sklearn.metrics import confusion_matrix #confusion matrix for checking accuracy

# Input data files are available in the "../input/" directory.
train_dataset = pd.read_csv('./../input/train.csv')  # training data
test_dataset = pd.read_csv('./../input/test.csv') #test data
#getting initial few record of test dataset
test_dataset.head()
#getting initial few record of training dataset
train_dataset.head()
#printing training and test columns which represt feature set.
print("train_dataset.columns ==>",train_dataset.columns)
print("test_dataset.columns ==>",test_dataset.columns)
# describe perform basic mathematical calculation over numerical data
train_dataset.describe()
test_dataset.describe()
# using include['O'] option will give categorical statistic.
train_dataset.describe(include=['O'])
test_dataset.describe(include=['O'])
# info of training and test data set.
train_dataset.info()
print("-"*40)
test_dataset.info()
# average of survived on Pclass which is [1,2,3]
print(train_dataset[['Pclass', 'Survived']].groupby('Pclass').mean())
sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train_dataset)
# survival based on sex
print(train_dataset[['Sex', 'Survived']].groupby('Sex').mean())
sns.barplot(x='Sex',y='Survived',data=train_dataset)
print(train_dataset[['SibSp', 'Survived']].groupby('SibSp').mean().sort_values(by='Survived',ascending=False))
sns.pointplot(x='SibSp',y='Survived',data=train_dataset)
# based on Embarked survival rate
print(train_dataset[['Embarked', 'Survived']].groupby('Embarked').mean())
sns.boxplot(x='Embarked',y='Age',data=train_dataset)
print(train_dataset[["Parch", "Survived"]].groupby('Parch').mean().sort_values(by='Survived',ascending=False))
g = sns.FacetGrid(train_dataset,col='Survived',aspect=4)
g.map(sns.distplot,'Age',bins=50)
g = sns.FacetGrid(train_dataset,col='Survived',row='Pclass', height=2.2, aspect=3)
g.map(plt.hist,'Age',alpha=.5,bins=50)
grid = sns.FacetGrid(data=train_dataset, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
#dropping ticket as it has a more missing values.
train_dataset.drop('Ticket',axis=1,inplace=True)
test_dataset.drop('Ticket', axis=1,inplace=True)
#after dropping ticket column checking dataset
train_dataset.head()
train_dataset['Name'].head()
#extracting Title from each name and storing in new Title column in dataset.
train_dataset['Title'] = train_dataset.Name.str.extract('([A-Za-z]+)\.')
# replacing less frequently columns as Rare.
train_dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare',inplace=True)

test_dataset['Title'] = test_dataset.Name.str.extract('([A-Za-z]+)\.')
test_dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare',inplace=True)

#grouping by title and summing survives
train_dataset.groupby('Title').sum()['Survived']
#replacing Mlle,Mme and Ms to Miss as they all are female only.
train_dataset['Title'] = train_dataset['Title'].replace('Mlle','Miss')
train_dataset['Title'] = train_dataset['Title'].replace('Mme','Miss')
train_dataset['Title'] = train_dataset['Title'].replace('Ms','Miss')

test_dataset['Title'] = test_dataset['Title'].replace('Mlle','Miss')
test_dataset['Title'] = test_dataset['Title'].replace('Mme','Miss')
test_dataset['Title'] = test_dataset['Title'].replace('Ms','Miss')

train_dataset.groupby('Title').sum()['Survived']
#checking unique value of embarked 
train_dataset['Embarked'].value_counts()
#check null value if any present in embarked feature.
train_dataset[train_dataset['Embarked'].isnull()]
#based on embarked fare on each Pclass, which we will see for our fare 80
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train_dataset)
#checking avg fare of pclass = 1 and survived which is equivalent to our data
train_dataset[(train_dataset['Pclass'] == 1) & (train_dataset['Survived']==1)].mean()['Fare']
train_dataset.groupby(['Embarked','Pclass']).mean()
# filling C in missing columns of Embarked feature.
train_dataset['Embarked'] = train_dataset['Embarked'].fillna('C')
train_dataset['Embarked'].isnull().sum()
#test_dataset['Embarked'].isnull().sum() 0 for test as well
train_dataset.drop('Cabin',axis=1,inplace=True)
test_dataset.drop('Cabin',axis=1,inplace=True)
#checking age null values
train_dataset['Age'].isnull().sum()
# filling age randomly between mean()-std() to mean()+std() of age
rand_age = np.random.randint(train_dataset.Age.mean()-train_dataset.Age.std(),train_dataset.Age.mean()+train_dataset.Age.std())
train_dataset.Age = train_dataset.Age.fillna(rand_age)
rand_test_age = np.random.randint(test_dataset.Age.mean()-test_dataset.Age.std(),test_dataset.Age.mean()+test_dataset.Age.std())
test_dataset.Age = test_dataset.Age.fillna(rand_age)
train_dataset['Age'].isnull().sum()
test_dataset['Age'].isnull().sum()
# creating age band column by different age column for different range of ages.
train_dataset['Ageband'] = pd.cut(train_dataset['Age'], 5)
train_dataset.head()
test_dataset['Ageband'] = pd.cut(test_dataset['Age'], 5)
test_dataset.head()
#printing age band unique columns
print(train_dataset['Ageband'].value_counts())
print("---"*40)
print(test_dataset['Ageband'].value_counts())
#converting age band into categorical value.
train_dataset.loc[train_dataset['Age']<= 16,'Age']=0
train_dataset.loc[(train_dataset['Age']>16) & (train_dataset['Age']<=32),'Age']=1
train_dataset.loc[(train_dataset['Age']>32) & (train_dataset['Age']<=48),'Age']=2
train_dataset.loc[(train_dataset['Age']>48) & (train_dataset['Age']<=64),'Age']=3
train_dataset.loc[(train_dataset['Age']>64),'Age']=4
train_dataset.head()
# converting float value of age into integer.
train_dataset.Age = train_dataset.Age.astype(int)
# converting test data as well into age categorical value.

test_dataset.loc[test_dataset['Age']<= 16,'Age']=0
test_dataset.loc[(test_dataset['Age']>16) & (test_dataset['Age']<=32),'Age']=1
test_dataset.loc[(test_dataset['Age']>32) & (test_dataset['Age']<=48),'Age']=2
test_dataset.loc[(test_dataset['Age']>48) & (test_dataset['Age']<=64),'Age']=3
test_dataset.loc[(test_dataset['Age']>64),'Age']=4
test_dataset.head()
# converting age into integer.
test_dataset.Age = test_dataset.Age.astype(int)
#generate family from SibSp and Parch...
train_dataset['Family']= train_dataset['SibSp']+train_dataset['Parch']+1
train_dataset.head()
test_dataset['Family']= test_dataset['SibSp']+test_dataset['Parch']+1
test_dataset.head()
# creating IsAlone column based on Family columns
train_dataset['IsAlone']=0
train_dataset.loc[train_dataset.Family == 1,'IsAlone']=1
test_dataset['IsAlone']=0
test_dataset.loc[test_dataset.Family == 1,'IsAlone']=1
# dividing fare into 4 fare band
train_dataset['Fareband'] = pd.qcut(train_dataset['Fare'],4)
train_dataset.head(10)
test_dataset['Fareband'] = pd.qcut(test_dataset['Fare'],4)
test_dataset.head(10)
# unique value of fare band..
print(train_dataset['Fareband'].value_counts())
print("---"*40)
print(test_dataset['Fareband'].value_counts())
# converting fare band into categorical numerical values
train_dataset.loc[train_dataset['Fare']<=7.91,'Fare']=0
train_dataset.loc[(train_dataset['Fare']>7.91) & (train_dataset['Fare']<=14.454),'Fare']=1
train_dataset.loc[(train_dataset['Fare']>14.454) & (train_dataset['Fare']<=31),'Fare']=2
train_dataset.loc[(train_dataset['Fare']>31),'Fare']=3
train_dataset.Fare = train_dataset.Fare.astype(int)

test_dataset.Fare = test_dataset.Fare.fillna(test_dataset.Fare.mean())
test_dataset.loc[test_dataset['Fare']<=7.91,'Fare']=0
test_dataset.loc[(test_dataset['Fare']>7.91) & (test_dataset['Fare']<=14.454),'Fare']=1
test_dataset.loc[(test_dataset['Fare']>14.454) & (test_dataset['Fare']<=31),'Fare']=2
test_dataset.loc[(test_dataset['Fare']>31),'Fare']=3
test_dataset.Fare = test_dataset.Fare.astype(int)


# dropping unwanted feature
train_dataset.columns
test_dataset.columns
train_dataset.drop(['Name','SibSp','Parch','Ageband','Family','Fareband'],axis=1,inplace=True)
test_dataset.drop(['Name','SibSp','Parch','Ageband','Family','Fareband'],axis=1,inplace=True)
# save passenger id as we need at end for submission
train_pid = train_dataset['PassengerId']
train_dataset.drop('PassengerId',axis=1,inplace=True)
test_pid = test_dataset['PassengerId']
test_dataset.drop('PassengerId',axis=1,inplace=True)
train_dataset.head()
# encoding text value into numerical value..
label_encoder = preprocessing.LabelEncoder()
train_dataset['Title'] = label_encoder.fit_transform(train_dataset['Title'])
train_dataset['Sex'] = label_encoder.fit_transform(train_dataset['Sex'])
train_dataset['Embarked'] = label_encoder.fit_transform(train_dataset['Embarked'])
train_dataset.head()
label_encoder = preprocessing.LabelEncoder()
test_dataset['Title'] = label_encoder.fit_transform(test_dataset['Title'])
test_dataset['Sex'] = label_encoder.fit_transform(test_dataset['Sex'])
test_dataset['Embarked'] = label_encoder.fit_transform(test_dataset['Embarked'])
test_dataset.head()
# drop survive from data set as it is actual value which we will need for training and comparison.
X = train_dataset.drop('Survived',axis=1)
Y = train_dataset['Survived']
Z = test_dataset # this we will use later... after all for checking prediction and cross validation
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)
# using Logistic Classifier
log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)
y_log_pred = log_reg.predict(X_test)
acc_log = round(log_reg.score(X_train, Y_train) * 100, 2)
print("accuracy score is = {0}\nf-score is = {1}".format(acc_log,f1_score(Y_test,y_log_pred)))
Y_test.count()
sns.heatmap(confusion_matrix(Y_test,y_log_pred),annot=True,fmt='2.0f')
accuracy_score = []
for i in range(1,10):
    svc = SVC(C=i,gamma='auto')
    svc.fit(X_train, Y_train)
    y_svc_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
    accuracy_score.append([i,acc_svc])
    
acc_df = pd.DataFrame(data=accuracy_score,columns=['C','accuracy'])
acc_df.plot(kind='barh', stacked=True)
acc_df.sort_values(by='accuracy',ascending=False)

# checking F1 score and confusion matrix for best c value SVC(C=4)
svc = SVC(C=4,gamma='auto')
svc.fit(X_train, Y_train)
y_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("accuracy score is = {0}\nf-score is = {1}".format(acc_svc,f1_score(Y_test,y_svc)))
sns.heatmap(confusion_matrix(Y_test,y_svc),annot=True,fmt='2.0f')
accuracy_kscore = []
for k in range(3,10):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
    accuracy_kscore.append([k,acc_knn])

acc_knndf = pd.DataFrame(data=accuracy_kscore,columns=['k','accuracy'])
sns.barplot(x=acc_knndf['k'],y=acc_knndf['accuracy'],data=acc_knndf)
acc_knndf.sort_values(by='accuracy',ascending=False)

# checking F1 score and confusion matrix for best c value SVC(C=4)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("accuracy score is = {0}\nf-score is = {1}".format(acc_knn,f1_score(Y_test,Y_knn)))
sns.heatmap(confusion_matrix(Y_test,Y_knn),annot=True,fmt='2.0f')
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_gauss = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
print("accuracy score is = {0}\nf-score is = {1}".format(acc_gaussian,f1_score(Y_test,Y_gauss)))
sns.heatmap(confusion_matrix(Y_test,Y_gauss),annot=True,fmt='2.0f')
decision_tree = DecisionTreeClassifier(max_depth=10)
decision_tree.fit(X_train, Y_train)
Y_decision = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
print("accuracy score is = {0}\nf-score is = {1}".format(acc_decision_tree,f1_score(Y_test,Y_decision)))
sns.heatmap(confusion_matrix(Y_test,Y_decision),annot=True,fmt='2.0f')
random_forest = RandomForestClassifier(n_estimators=400)
random_forest.fit(X_train, Y_train)
Y_random = random_forest.predict(X_test)
# we will use at end of submission
Y_Pred = random_forest.predict(Z)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
#confusion_matrix(Y_train,Y_pred)
print("accuracy score is = {0}\nf-score is = {1}".format(acc_random_forest,f1_score(Y_test,Y_random)))
sns.heatmap(confusion_matrix(Y_test,Y_random),annot=True,fmt='2.0f')
ratios = random_forest.feature_importances_
feature_important = pd.DataFrame(index=X_train.columns, data=ratios, columns=['importance'])
feature_important
feature_important = feature_important.sort_values(by=['importance'], ascending=True)
feature_important.plot(kind='barh', stacked=True, color=['cornflowerblue'], grid=False, figsize=(8, 5))
from sklearn.model_selection import KFold   # k-fold 
from sklearn.model_selection import cross_val_score #cross- validation scrore
from sklearn.model_selection import cross_val_predict

kfold = KFold(n_splits=10,random_state=10)

classfy_mean = [];
accuracy_kfold = []
# list of all classifier
classifier = ['Logistic regression','SVC','KNN','Gaussian NB','Decision Tree','Random Forest']

#list of classifier model
models = [LogisticRegression(),SVC(C=4,gamma='auto'),KNeighborsClassifier(n_neighbors=3),GaussianNB(),
          DecisionTreeClassifier(max_depth=10),RandomForestClassifier(n_estimators=100)]

for i in models:
    model = i
    cv_result = cross_val_score(model,X,Y,cv=kfold,scoring='accuracy')
    accuracy_kfold.append(cv_result)
    classfy_mean.append(cv_result.mean())

classfy_mean
#accuracy_kfold
df = pd.DataFrame(data=classfy_mean,index=classifier,columns=['cross-accuracy'])
df.sort_values(by='cross-accuracy',ascending=False)
sns.pointplot(x=df.index,y=df['cross-accuracy'])
# save passenger id as we need at end for submission
submission = pd.DataFrame({
        "PassengerId": test_pid,
        "Survived": Y_Pred
})
submission
# len(test_pid)
# len(Y_Pred)
submission.to_csv("titanic_submission.csv", index=False)
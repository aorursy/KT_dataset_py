import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
%matplotlib inline
# Below two commands are used to fetch data from CSV file.
train_dataset_raw = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')

# We created a duplicate copy of train dataset to manipulate it. 
train_dataset = train_dataset_raw.copy(deep = True)

# Here the train and test dataset are stored in a list so that we dont have to manipulate both one by one
data_cleaner = [train_dataset, test_dataset]

# Displays the overall details about the train dataset.
train_dataset_raw.describe(include = 'all')
print(train_dataset.isnull().sum())
print ('-'*10)
print (test_dataset.isnull().sum())
# This loop will manipulate both train and test dataset.
for dataset in data_cleaner:    
    #complete missing age values with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:    
        
    #creates a new column 'CabinBool' which replaces the null values with '0' and who has cabin with '1'.
    dataset['CabinBool'] = (dataset['Cabin'].notnull().astype('int'))
    
    #Discrete variables - creates a family size with no. of siblings , spouse , parents or children.
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    #This will store the title of each person from its 'Name'.
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    #This will create a Fare Slab according to no. provided. In our case we want '4' Fare Slab.
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    #Same goes with here, we will have 6 Age categories.
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 6)
# Check our training data after transformation.
train_dataset.sample(10)
print(train_dataset['Title'].value_counts())
#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (train_dataset['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
train_dataset['Title'] = train_dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(train_dataset['Title'].value_counts())
print("-"*10)
title_namest = (test_dataset['Title'].value_counts() < stat_min)
test_dataset['Title'] = test_dataset['Title'].apply(lambda x: 'Misc' if title_namest.loc[x] == True else x)
label = LabelEncoder()
for dataset in data_cleaner:
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    
print(train_dataset.columns)
test_dataset.columns
sns.heatmap(train_dataset.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
features_col=['Pclass','Sex','Embarked','IsAlone','Title','AgeBin_Code','FamilySize','FareBin_Code','CabinBool']
train_ds = train_dataset[features_col]
test_ds = test_dataset[features_col]
train_label = train_dataset['Survived']
print(train_ds.columns)
train_ds.head()
one_hot_encoded_training_predictors = pd.get_dummies(train_ds)
one_hot_encoded_testing_predictors = pd.get_dummies(test_ds)
sns.heatmap(one_hot_encoded_training_predictors.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()
# Remove the correlated feature to reduce redundancy in model.
corln_col=['Title_Miss','Sex_male']
one_hot_encoded_training_predictors = one_hot_encoded_training_predictors.drop(corln_col,axis=1)
one_hot_encoded_testing_predictors = one_hot_encoded_testing_predictors.drop(corln_col,axis=1)
# Import all the required packages.
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import svm #support vector Machine
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(one_hot_encoded_training_predictors, train_label, test_size=0.20)

svc_clf = SVC(kernel='rbf',C=1,gamma=0.1) #we can try different parameters
svc_clf.fit(train_X, train_y)
pred_svc = svc_clf.predict(test_X)
acc_svc = accuracy_score(test_y, pred_svc)

print(acc_svc)
linsvc_clf = LinearSVC()
linsvc_clf.fit(train_X, train_y)
pred_linsvc = linsvc_clf.predict(test_X)
acc_linsvc = accuracy_score(test_y, pred_linsvc)

print(acc_linsvc)
logreg_clf = LogisticRegression()
logreg_clf.fit(train_X, train_y)
pred_logreg = logreg_clf.predict(test_X)
acc_logreg = accuracy_score(test_y, pred_logreg)

print(acc_logreg)
knn_clf = KNeighborsClassifier()
knn_clf.fit(train_X, train_y)
pred_knn = knn_clf.predict(test_X)
acc_knn = accuracy_score(test_y, pred_knn)

print(acc_knn)

gnb_clf = GaussianNB()
gnb_clf.fit(train_X, train_y)
pred_gnb = gnb_clf.predict(test_X)
acc_gnb = accuracy_score(test_y, pred_gnb)

print(acc_gnb)
#Compare all model performance.
model_performance = pd.DataFrame({
    'Model': ['Radial SVM', 'Linear SVM', 
              'Logistic Regression', 'K Nearest Neighbors', 'Gaussian Naive Bayes'],
    'Accuracy': [acc_svc, acc_linsvc, 
              acc_logreg, acc_knn, acc_gnb]
})

model_performance.sort_values(by='Accuracy', ascending=False)
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear',gamma='auto'),svm.SVC(kernel='rbf',gamma='auto'),LogisticRegression(),KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),RandomForestClassifier(n_estimators=100)]
for i in models:
    model = i
    cv_result = cross_val_score(model, train_X, train_y, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2
sub_clf = RandomForestClassifier(n_estimators=100)
sub_clf.fit(one_hot_encoded_training_predictors, train_label)
submission_predictions = sub_clf.predict(one_hot_encoded_testing_predictors)
submission = pd.DataFrame({
        "PassengerId": test_dataset['PassengerId'],
        "Survived": submission_predictions
    })

submission.to_csv("titanic.csv", index=False)
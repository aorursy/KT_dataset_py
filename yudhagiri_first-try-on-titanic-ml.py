# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
cf.go_offline()
sns.set_style('darkgrid')
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
#train data head
train.head()
#test data head
test.head()
#data info
train.info()
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
#percentage of missing Age and Cabin Data
age_miss = ((train[train['Age'].isnull()==True].index.size)/train.index.size)*100
print("Missing Age Data: " + str(round(float(age_miss), 1)) + "%")
cabin_miss = ((train[train['Cabin'].isnull()==True].index.size)/train.index.size)*100
print("Missing Cabin Data: " + str(round(float(cabin_miss),1)) + "%")
train = train.drop('Cabin', axis=1)
train.head()
plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass', y='Age', data=train)
#finding mean age for each class
first_mean = round(train[train['Pclass']==1]['Age'].mean())
sec_mean = round(train[train['Pclass']==2]['Age'].mean())
third_mean = round(train[train['Pclass']==3]['Age'].mean())
print("First class age mean: " + str(first_mean))
print("Second class age mean: " + str(sec_mean))
print("Third class age mean: " + str(third_mean))
#actually filling up the unknown Age value according to age mean for each class

def fill_age(col):
    Age = col[0]
    Pclass = col[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return first_mean
        elif Pclass == 2:
            return sec_mean
        else:
            return third_mean
    
    else:
        return Age
train['Age'] = train[['Age', 'Pclass']].apply(fill_age, axis=1)
#confirm no more missing data
train[train.isnull()==True].index
sns.heatmap(train.isnull()==True, yticklabels=False)
def age_group(col):
    Age = col[0]
    
    if Age<13:
        return 1
    elif 13<= Age <= 18:
        return 2
    elif 19<= Age <= 27:
        return 3
    elif 27<= Age <=50:
        return 4
    else:
        return 5
train['age_group'] = train[['Age']].apply(age_group, axis=1)

train.head()
sex = pd.get_dummies(train['Sex'])
sex.head()
#reducing column to male only, therefore 1 is male 0 is female, this is to reduce data needed
sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()
#combining this categorical feature with train dataframe
train = pd.concat([train, sex], axis=1)
train = train.drop('Sex', axis=1)
train.rename(columns ={'male': 'gender'}, inplace=True)
train.head(10)
#categorical feature from passenger title
titles = ['Miss.', 'Mr.', 'Mrs.', 'Master.']

def find_title(col):
    Name = col[0]
    
    passname = str(col).split()
    
    if titles[0] in passname:
        return 1
    elif titles[1] in passname:
        return 2
    elif titles[2] in passname:
        return 3
    elif titles[3] in passname:
        return 4
    else:
        return 5
train['titles_group'] = train[['Name']].apply(find_title, axis=1)
train.head()
def embark(col):
    embarked = col[0]
    
    if embarked == 'S':
        return 1
    elif embarked == 'C':
        return 2
    else:
        return 3
train['embark_group'] = train[['Embarked']].apply(embark, axis=1)
train.head()
train = train.drop(['Name', 'Embarked', 'Ticket'], axis=1)
train.head()
train['Fare'] = train['Fare'].astype(int)
train['Fare'].unique()
train['Fare'].mean()
print ('Q1: '  + str(train['Fare'].quantile(0.25)))
print ('median: ' + str(train['Fare'].median()))
print ('Q3: ' + str(train['Fare'].quantile(0.75)))
plt.figure(figsize=(8,6))
sns.distplot(train['Fare'], kde=False, bins=50)
def fare_group(col):
    fare = col[0]
    
    if fare <= 7:
        return 1
    elif 7< fare <=15:
        return 2
    elif 15< fare <= 31:
        return 3
    elif 31< fare <= 99:
        return 4
    elif 99< fare <=200:
        return 5
    else:
        return 6
    
train['fare_group'] = train[['Fare']].apply(fare_group, axis=1)
train.head()
train = train.drop('Fare', axis=1)
train.head()
tcor = train.drop('PassengerId', axis=1).corr()
plt.figure(figsize=(10,6))
sns.heatmap(tcor, annot=True, cmap ='coolwarm')
test = pd.read_csv('../input/titanic/test.csv')
test.head()
sns.heatmap(test.isnull(), yticklabels=False, cbar=False)
#drop cabin
test = test.drop('Cabin', axis=1)
test.head()
#fill unknown age values and age-grouping
test['Age'] = test[['Age', 'Pclass']].apply(fill_age, axis=1)
test['age_group'] = test[['Age']].apply(age_group, axis=1)
test.head()
#sex categorizing
t_sex = pd.get_dummies(test['Sex'], drop_first=True)
t_sex.head()
test = pd.concat([test, t_sex], axis=1)
test = test.drop('Sex', axis=1)
test.rename(columns ={'male': 'gender'}, inplace=True)
test.head()
#title grouping
test['titles_group'] = test[['Name']].apply(find_title, axis=1)
test.head(2)
#embark grouping
test['embark_group'] = test[['Embarked']].apply(embark, axis=1)
#fare grouping
test['fare_group'] = test[['Fare']].apply(fare_group, axis=1)
test.head()
test = test.drop(['Name', 'Embarked', 'Ticket', 'Fare'], axis=1)
test.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#check data one more time
train.head()
test.head()
X_train = train.drop(['PassengerId', 'Survived'], axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId'], axis=1)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
log_pred = logmodel.predict(X_test)
log_pred
log_model_pred = pd.DataFrame(log_pred, columns=['log_Survived'])
log_model_pred.head()
log_model_pred.value_counts()
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)
dtree_model_pred=pd.DataFrame(dtree_pred, columns=['dtree_Survived'])
dtree_model_pred.value_counts()
rtree = RandomForestClassifier(n_estimators=500)
rtree.fit(X_train, y_train)
rtree_pred = rtree.predict(X_test)
rtree_model_pred = pd.DataFrame(rtree_pred, columns=['rtree_Survived'])
rtree_model_pred.value_counts()
#joining all result for better view
survivor_data = pd.concat([log_model_pred, dtree_model_pred, rtree_model_pred], axis=1)
survivor_data.head()
#joining with test data
test_fitted = pd.concat([test, survivor_data], axis=1)
test_fitted.head()
#finding where the difference between all model occur
test_fitted[(test_fitted['log_Survived'] != test_fitted['dtree_Survived']) | 
            (test_fitted['log_Survived'] != test_fitted['rtree_Survived']) |
            (test_fitted['dtree_Survived'] != test_fitted['rtree_Survived'])].count()
test_fitted[(test_fitted['log_Survived'] != test_fitted['dtree_Survived']) | 
            (test_fitted['log_Survived'] != test_fitted['rtree_Survived']) |
            (test_fitted['dtree_Survived'] != test_fitted['rtree_Survived'])].index
test_fitted[test_fitted['log_Survived'] != test_fitted['dtree_Survived']].count()
test_fitted[test_fitted['log_Survived'] != test_fitted['rtree_Survived']].count()
test_fitted[test_fitted['dtree_Survived'] != test_fitted['rtree_Survived']].count()
print("Number of unique values of age in prediction difference: ")
print(str(test_fitted[(test_fitted['log_Survived'] != test_fitted['dtree_Survived']) | 
            (test_fitted['log_Survived'] != test_fitted['rtree_Survived']) |
            (test_fitted['dtree_Survived'] != test_fitted['rtree_Survived'])]['Age'].nunique()))
print("Number of unique values of fare group in prediction difference: ")
print(str(test_fitted[(test_fitted['log_Survived'] != test_fitted['dtree_Survived']) | 
            (test_fitted['log_Survived'] != test_fitted['rtree_Survived']) |
            (test_fitted['dtree_Survived'] != test_fitted['rtree_Survived'])]['fare_group'].nunique()))
print("Number of unique values of titles_group in prediction difference: ")
print(str(test_fitted[(test_fitted['log_Survived'] != test_fitted['dtree_Survived']) | 
            (test_fitted['log_Survived'] != test_fitted['rtree_Survived']) |
            (test_fitted['dtree_Survived'] != test_fitted['rtree_Survived'])]['titles_group'].nunique()))
X_train_new= train.drop(['PassengerId', 'Age', 'Survived'], axis=1)
y_train_new = train['Survived']
X_test_new = test.drop(['PassengerId', 'Age'], axis=1)
log2 = LogisticRegression()
log2.fit(X_train_new, y_train_new)
log_pred2 = log2.predict(X_test_new)
dtree2 = DecisionTreeClassifier()
dtree2.fit(X_train_new, y_train_new)
dtree_pred2 = dtree2.predict(X_test_new)
rtree2 = RandomForestClassifier()
rtree2.fit(X_train_new, y_train_new)
rtree_pred2 = rtree2.predict(X_test_new)
log_model_new = pd.DataFrame(log_pred2, columns=['log_Survived_new'])
dtree_model_new = pd.DataFrame(dtree_pred2, columns=['dtree_Survived_new'])
rtree_model_new = pd.DataFrame(rtree_pred2, columns=['rtree_Survived_new'])
survivor_new = pd.concat([log_model_new, dtree_model_new, rtree_model_new], axis=1)
survivor_new.head()
test_fitted_new = pd.concat([test, survivor_new], axis=1)
test_fitted_new.head()
test_fitted_new[(test_fitted_new['log_Survived_new'] != test_fitted_new['dtree_Survived_new']) | 
            (test_fitted_new['log_Survived_new'] != test_fitted_new['rtree_Survived_new']) |
            (test_fitted_new['dtree_Survived_new'] != test_fitted_new['rtree_Survived_new'])].count()
print("Number of unique values of age in prediction difference: ")
print( str(test_fitted_new[(test_fitted_new['log_Survived_new'] != test_fitted_new['dtree_Survived_new']) | 
            (test_fitted_new['log_Survived_new'] != test_fitted_new['rtree_Survived_new']) |
            (test_fitted_new['dtree_Survived_new'] != test_fitted_new['rtree_Survived_new'])]['Age'].nunique()))
acc_log_model = round(logmodel.score(X_train, y_train)*100 ,2)
acc_log2_model = round(log2.score(X_train_new, y_train_new)*100,2)
acc_dtree_model = round(dtree.score(X_train, y_train)*100,2)
acc_dtree2_model = round(dtree2.score(X_train_new, y_train_new)*100,2)
acc_rtree_model = round(rtree.score(X_train, y_train)*100,2)
acc_rtree2_model = round(rtree2.score(X_train_new, y_train_new)*100,2)


print('First Modelling Acc: \n')
print('Logistic Regression:  ' + str(acc_log_model) +"%")
print('Decision Tree: ' + str(acc_dtree_model) +"%")
print('RandomForest: ' + str(acc_rtree_model) +"%")
print('\nSecond Modelling Acc: \n')
print('Logistic Regression:  ' + str(acc_log2_model) +"%")
print('Decision Tree: ' + str(acc_dtree2_model) +"%")
print('RandomForest: ' + str(acc_rtree2_model) +"%")
fig,axes= plt.subplots(3,2,figsize=(10,8))

sns.countplot('age_group',data=test_fitted, hue='log_Survived', ax=axes[0,0])
sns.countplot('age_group',data=test_fitted, hue='dtree_Survived', ax=axes[1,0])
sns.countplot('age_group',data=test_fitted, hue='rtree_Survived', ax=axes[2,0])
axes[0,0].set_title('Age Group Survival Chance (1st Model)')

sns.countplot('age_group',data=test_fitted_new, hue='log_Survived_new', ax=axes[0,1])
sns.countplot('age_group',data=test_fitted_new, hue='dtree_Survived_new', ax=axes[1,1])
sns.countplot('age_group',data=test_fitted_new, hue='rtree_Survived_new', ax=axes[2,1])
axes[0,1].set_title('Age Group Survival Chance (2nd Model)')

plt.tight_layout()
fig,axes= plt.subplots(3,2,figsize=(10,10))

sns.countplot('Pclass',data=test_fitted, hue='log_Survived', ax=axes[0,0])
sns.countplot('Pclass',data=test_fitted, hue='dtree_Survived', ax=axes[1,0])
sns.countplot('Pclass',data=test_fitted, hue='rtree_Survived', ax=axes[2,0])
axes[0,0].set_title('Passenger Class Survival Chance (1st Model)')

sns.countplot('Pclass',data=test_fitted_new, hue='log_Survived_new', ax=axes[0,1])
sns.countplot('Pclass',data=test_fitted_new, hue='dtree_Survived_new', ax=axes[1,1])
sns.countplot('Pclass',data=test_fitted_new, hue='rtree_Survived_new', ax=axes[2,1])
axes[0,1].set_title('Passenger Class Survival Chance (2nd Model)')

plt.tight_layout()
fig,axes= plt.subplots(3,2,figsize=(10,10))

sns.countplot('gender',data=test_fitted, hue='log_Survived', ax=axes[0,0])
sns.countplot('gender',data=test_fitted, hue='dtree_Survived', ax=axes[1,0])
sns.countplot('gender',data=test_fitted, hue='rtree_Survived', ax=axes[2,0])
axes[0,0].set_title('Passenger Gender Survival Chance (1st Model)')

sns.countplot('gender',data=test_fitted_new, hue='log_Survived_new', ax=axes[0,1])
sns.countplot('gender',data=test_fitted_new, hue='dtree_Survived_new', ax=axes[1,1])
sns.countplot('gender',data=test_fitted_new, hue='rtree_Survived_new', ax=axes[2,1])
axes[0,1].set_title('Passenger Gender Survival Chance (2nd Model)')

plt.tight_layout()
#figuring out column importance order
feature = pd.DataFrame({'feature' : X_train.columns, 
                       'importance': np.round(rtree.feature_importances_, 2)})
feature_rank = feature.sort_values('importance', ascending=False)
plt.figure(figsize=(8,6))
sns.barplot(y = 'feature', x = 'importance' , data=feature_rank)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict
predict = cross_val_predict(rtree, X_train, y_train)
print('Confusion matrix: \n')
print(confusion_matrix(y_train, predict))
print('\nClassification report:\n')
print(classification_report(y_train, predict))
submit_file = test_fitted[['PassengerId', 'rtree_Survived']]
submit_file.rename(columns={'rtree_Survived' : 'Survived'}, inplace=True)
submit_file.head(15)
submit_file.to_csv('submission.csv', index=False)
train.head()
test.head()
def is_alone(cols):
    SibSp =cols[0]
    Parch = cols[1]
    
    if SibSp == 0 and Parch == 0:
        return 1
    else: 
        return 0
train['is_alone'] = train[['SibSp', 'Parch']].apply(is_alone, axis=1)
train = train.drop(['SibSp', 'Parch'], axis=1)
train.head()
test['is_alone'] = test[['SibSp', 'Parch']].apply(is_alone, axis =1)
test = test.drop(['SibSp', 'Parch'], axis=1)
test.head()
#more category column of age*class
def age_class(cols):
    Pclass= cols[0]
    Age = cols[1]
    
    return Pclass*Age
train['age_class'] = train[['Pclass', 'age_group']].apply(age_class, axis=1)
test['age_class'] = test[['Pclass', 'age_group']].apply(age_class, axis=1)
train.head()
test.head()
#modeling using RandomForest

X_train = train.drop(['PassengerId', 'Survived', 'Age'], axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId', 'Age'], axis=1)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest_acc = round(random_forest.score(X_train, y_train)*100,2)
predict = cross_val_predict(rtree, X_train, y_train)

print('Random Forest Updated Acc: ' + str(random_forest_acc))
print(classification_report(y_train, predict))
random_forest_model = pd.DataFrame({
    'Survived': y_pred
})
survivor_updated = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : y_pred
})
survivor_updated.head(15)
survivor_updated.to_csv('second_submit.csv', index=False)
#modeling using RandomForest

X_train = train.drop(['PassengerId', 'Survived', 'age_group', 'Age'], axis=1)
y_train = train['Survived']
X_test = test.drop(['PassengerId', 'age_group', 'Age'], axis=1)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred_third = random_forest.predict(X_test)
random_forest_acc = round(random_forest.score(X_train, y_train)*100,2)
predict = cross_val_predict(rtree, X_train, y_train)

print('Random Forest Updated Acc: ' + str(random_forest_acc))
survivor_updated_again = pd.DataFrame({
    'PassengerId' : test['PassengerId'],
    'Survived' : y_pred_third
})
survivor_updated_again.head(15)
survivor_updated_again.to_csv('third_submit.csv', index=False)

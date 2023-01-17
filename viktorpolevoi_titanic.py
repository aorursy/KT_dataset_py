import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid", palette="Paired")
plt.rcParams['figure.dpi'] = 120
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
titanic = pd.read_csv('../input/titanic/train.csv', index_col = 'PassengerId')
titanic_test = pd.read_csv('../input/titanic/test.csv', index_col = 'PassengerId')
titanic
titanic.info()
titanic.describe()
surv_count = titanic.Survived.value_counts()
surv_pie = plt.pie(titanic.Survived.value_counts(), 
                   labels=[f'Not survived: {surv_count[0]}', f'Survived: {surv_count[1]}'], 
                   autopct='%1.1f%%', startangle=90)
plt.figure(figsize=(7,5))
gender_surv = sns.countplot(x="Survived", hue="Sex", data=titanic)
gender_surv.set_xticklabels(['Not survived', 'Survived'])
for p in gender_surv.patches:
    gender_surv.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 5),
                         textcoords = 'offset points')
plt.xlabel('')
pd.crosstab(titanic['Survived'],titanic['Sex'])
titanic['Age_group'] = titanic.Age
titanic.Age_group.fillna(-1, inplace = True)
titanic.Age_group = titanic.Age_group.map(lambda age: int(age//10+1))
titanic.Age_group.value_counts()
titanic[titanic.Age >= 70]
age_group_pie_set = titanic.Age_group[titanic.Age_group != 0].value_counts()
explode = (0,0,0,0,0,0,0,1,0)
labels_group_age = [f'{x*10-10} - {x*10-1} y/o' for x in age_group_pie_set.index]
plt.figure(figsize=(9,7))
age_group_pie = plt.pie(age_group_pie_set, explode=explode, labels=labels_group_age, autopct='%1.1f%%')
by_gender = pd.pivot_table(titanic, values = 'Age', index=['Survived', 'Sex'],
                     columns=['Age_group'], fill_value=0, aggfunc=len)
by_gender.drop(0, axis=1, inplace=True)
by_gender
age_group_surv = titanic.groupby(['Age_group', 'Survived'])
age_group_surv = age_group_surv.size().unstack()
age_group_surv = age_group_surv[1:].fillna(0)
age_group_surv
age_group_surv_barh = age_group_surv[:7].plot(kind='barh')
plt.legend(['Not survived', 'Survived'])
for p in age_group_surv_barh.patches:
    age_group_surv_barh.annotate(int(p.get_width()), 
                        (p.get_y() + p.get_width()+2, 
                         p.get_height() + p.get_y()-0.3),
                         ha = 'center', 
                         va = 'center',
                         xytext = (2, 5),
                         textcoords = 'offset points')
age_group_surv[:7].apply(lambda x:x/x.sum(), axis=1).plot(kind='barh', stacked=True, legend=False)
plt.legend(['Not survived', 'Survived'], bbox_to_anchor=(1.0, 0.7))
sns.catplot(x = 'Sex', y = 'Age', hue = 'Survived',data=titanic, kind='swarm')
plt.figure(figsize=(7,5))
pcl_surv = sns.countplot(x='Survived', hue='Pclass',data=titanic)
pcl_surv.set_xticklabels(['Not survived', 'Survived'])
plt.xlabel('')
for p in pcl_surv.patches:
    pcl_surv.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 5),
                         textcoords = 'offset points')
plt.figure(figsize=(7,5))
embarked_surv = sns.countplot(x='Survived', hue='Embarked',data=titanic)
embarked_surv.set_xticklabels(['Not survived', 'Survived'])
plt.xlabel('')
for p in embarked_surv.patches:
    embarked_surv.annotate(p.get_height(), 
                        (p.get_x() + p.get_width() / 2.0, 
                         p.get_height()), 
                         ha = 'center', 
                         va = 'center', 
                         xytext = (0, 5),
                         textcoords = 'offset points')
sns.relplot(x = 'Age', y = 'Fare', style = 'Survived', color='steelblue', data = titanic)
titanic[titanic.Fare == 0]
titanic.Fare[titanic.Fare == 0] = titanic.Fare.median(axis = 0)
titanic.Age[titanic.Age.isnull()] = titanic.Age.median(axis = 0)
sns.heatmap(titanic.isnull(), yticklabels=False, cmap="viridis")
titanic.drop(["Cabin", "Name", "Ticket", "Age_group"], axis=1, inplace=True)
titanic.dropna(inplace=True)
titanic[:2]
pcl = pd.get_dummies(titanic["Pclass"],drop_first=True)
embark = pd.get_dummies(titanic["Embarked"])
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = titanic
a = titanic['Sex']

X['Sex'] = le.fit_transform(X['Sex'])

a = le.transform(a)
dataset = X
titanic = pd.concat([titanic,embark,pcl],axis=1)
titanic.head()
titanic.drop(['Pclass', 'Embarked'], axis=1, inplace = True)
titanic_corr = titanic.corr()
plt.figure(figsize=(13,5))
sns.heatmap(data=titanic_corr, annot=True, cmap='GnBu')
X = titanic.drop("Survived", axis=1)
y = titanic["Survived"]
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()
log_model.fit(X,y)
titanic_test
titanic_test.Age[titanic_test.Age.isnull()] = titanic_test.Age.median(axis = 0)
titanic_test.Fare[titanic_test.Fare == 0] = titanic_test.Fare.median(axis = 0)
titanic_test.Fare[titanic_test.Fare.isnull()] = titanic_test.Fare.median(axis = 0)
titanic_test.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)
pcl_t = pd.get_dummies(titanic_test["Pclass"],drop_first=True)
embark_t = pd.get_dummies(titanic_test["Embarked"])

X_t = titanic_test
a_t = titanic_test['Sex']

X_t['Sex'] = le.fit_transform(X_t['Sex'])

a = le.transform(a_t)
dataset_t = X_t
titanic_test = pd.concat([titanic_test,embark_t,pcl_t],axis=1)
titanic_test.head()
titanic_test.drop(['Pclass', 'Embarked'], axis=1, inplace = True)
predictions = log_model.predict(titanic_test)
submission = pd.DataFrame({
        "PassengerId": titanic_test.index,
        "Survived": predictions
    })
submission.to_csv('gender_submission.csv', index = False)
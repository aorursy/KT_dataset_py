# import useful and reqired libraries

import numpy as np
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) # check out what files we have
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
for sex in train['Sex']:
    if sex != 'male' and sex != 'female':
        print('ERROR!')
train['Sex'] = (train['Sex']=='male') * 1
test['Sex'] = (test['Sex']=='male') * 1
combine = pd.concat([train.drop('Survived',1),test])
train.head(10) # overview the training data set 
train.describe()
train.isnull().sum()  # to count missing values
survived = np.sum(train['Survived']==1)
total = len(train)
print("Survived: {} of {} ({} percents)".format(survived, total, round(survived * 100.00 /total, 2)))
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=[20,15])
plt.subplot(331)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(332)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(333)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(336)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(339)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(334)
surv_ages = train[train['Survived']==1]['Age'].dropna()
sns.distplot(surv_ages, bins=range(0, 81, 1), kde=False, color='green')
p = plt.ylabel('Survived')
plt.subplot(335)
notsurv_ages = train[train['Survived']==0]['Age'].dropna()
sns.distplot(notsurv_ages, bins=range(0, 81, 1), kde=False, color='red')
p = plt.ylabel('Not survived')
plt.subplot(337)
surv_fare = train[train['Survived']==1]['Fare'].dropna()
sns.distplot(np.log10(surv_fare + 1), kde=False, color='green')
p = plt.ylabel('Survived')
plt.subplot(338)
notsurv_fare = train[train['Survived']==0]['Fare'].dropna()
sns.distplot(np.log10(notsurv_fare + 1), kde=False, color='red')
p = plt.ylabel('Not survived')
print("Median ages for survived: {}, not survived: {}".format(np.median(surv_ages), np.median(notsurv_ages)))
print("Median fare for survived: {}, not survived: {}".format(np.median(surv_fare), np.median(notsurv_fare)))
tab_SibSp = pd.crosstab(train['SibSp'], train['Survived'])  # cross table to check the number of survived vs not survived passengers
tab_SibSp['survived, %'] = (tab_SibSp[1]/(tab_SibSp[0] + tab_SibSp[1])) * 100.0
print(tab_SibSp)
tab_Parch = pd.crosstab(train['Parch'], train['Survived'])  # cross table to check the number of survived vs not survived passengers
tab_Parch['survived, %'] = (tab_Parch[1]/(tab_Parch[0] + tab_Parch[1])) * 100.0
print(tab_Parch)
plt.figure(figsize=(14,12))
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), annot=True)
plt.figure(figsize=[20,20])
plt.subplot(221)
sns.violinplot(x="Pclass", y=np.log10(train["Fare"] + 1), hue="Survived", data=train, split=True)
plt.hlines([1, 1.5], xmin=-1, xmax=3, linestyles="dotted")
plt.subplot(222)
sns.violinplot(x="Parch", y="SibSp", hue="Survived", data=train, split=True)
plt.hlines([-1, 1], xmin=-1, xmax=6, linestyles="dotted")
plt.subplot(223)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True)
plt.hlines([10, 30, 50, 70], xmin=-1, xmax=3, linestyles="dotted")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked",
                   data=train, aspect=0.9, size=3.5, ci=95.0)
tab = pd.crosstab(combine['Embarked'], combine['Pclass'])
print(tab)
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
p = plt.ylabel('Percentage')
tab = pd.crosstab(combine['Parch'], combine['SibSp'])
print(tab)
sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train)
tab = pd.crosstab(combine['Embarked'], combine['Sex'])
print(tab)
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
p = plt.ylabel('Percentage')
sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True)
plt.hlines([10,30,50], xmin=-1, xmax=3, linestyles="dotted")
plt.figure(figsize=[20,15])
survived = train[train['Survived']==1]
not_survived = train[train['Survived']==0]
plt.subplot(311)
plt.title("Pclass = 1")
sns.distplot(np.log10(survived['Fare'][survived['Pclass']==1].dropna() + 1), kde=False, color='green')
sns.distplot(np.log10(not_survived['Fare'][not_survived['Pclass']==1].dropna() + 1), kde=False, color='red')
plt.subplot(312)
plt.title("Pclass = 2")
sns.distplot(np.log10(survived['Fare'][survived['Pclass']==2].dropna() + 1), kde=False, color='green')
sns.distplot(np.log10(not_survived['Fare'][not_survived['Pclass']==2].dropna() + 1), kde=False, color='red')
plt.subplot(313)
plt.title("Pclass = 3")
sns.distplot(np.log10(survived['Fare'][survived['Pclass']==3].dropna() + 1), kde=False, color='green')
sns.distplot(np.log10(not_survived['Fare'][not_survived['Pclass']==3].dropna() + 1), kde=False, color='red')
ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train);
ax.set_yscale('log')
print(train[train['Embarked'].isnull()])
# titanic %>% group_by(Pclass, Embarked) %>% summarise(count = n(), median_fare = median(Fare, na.rm=TRUE))
combine.where((combine['Sex'] == 0) & (combine['Parch'] == 0) & \
              (combine['SibSp'] == 0) & (combine['Pclass'] == 1) & \
              (combine['Fare'] > 50) & (combine['Fare'] < 110)).groupby(['Embarked']).size()
train.loc[61,'Embarked'] = "C"
train.loc[829,'Embarked'] = "C"
train.isnull().sum()  # to count missing values

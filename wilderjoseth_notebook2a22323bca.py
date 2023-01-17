# Tools for data exploration
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math
import scipy.stats as stats
dsTrain = pd.read_csv('../input/titanic/train.csv')
dsTest = pd.read_csv('../input/titanic/test.csv')
dsTrain.head()
dsTrain.shape
dsTrain.info()
dsTrain[['PassengerId','Age', 'Fare', 'SibSp', 'Parch']].describe()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12, 15))

sns.boxplot(x = dsTrain['Survived'], y = dsTrain['Age'], ax = ax1)
ax1.set_title('Boxplot')

ax2.hist(dsTrain['Age'], bins = 50)
ax2.set_title('Age distribution')

ax3.scatter(dsTrain['Age'], dsTrain['Survived'])
ax3.set_title('Age relation')

plt.show()
dsTrain[['Age', 'Survived']].groupby(['Survived']).agg(['mean', 'count', 'std']).reset_index()
# Calculating p-value
alpha = 0.05 #Significance level
nSurvived = dsTrain[dsTrain['Survived'] == 1]['Age'].count()
nNoSurvived = dsTrain[dsTrain['Survived'] == 0]['Age'].count()
varSurvived = dsTrain[dsTrain['Survived'] == 1]['Age'].var()
varNoSurvived = dsTrain[dsTrain['Survived'] == 0]['Age'].var()
meanSurvived = dsTrain[dsTrain['Survived'] == 1]['Age'].mean()
meanNoSurvived = dsTrain[dsTrain['Survived'] == 0]['Age'].mean()
diffMean = meanSurvived - meanNoSurvived
se = np.sqrt((varSurvived / nSurvived) + (varNoSurvived / nNoSurvived))

print('Null hypothesis:',  0)
print('Alternative hypothesis:',  diffMean)
print('Standard error:', se)

tTest = np.abs((diffMean - 0) / se) #remove negative result
print('t-Test:', tTest)

df = np.min([nSurvived - 1, nNoSurvived - 1])
print('Degrees of freedom:', df)

p_value = (1 - stats.t.cdf(tTest, df = df)) * 2 #Calculating two-sided hypothesis
print('p-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and I can say that there is a difference between survived mean and no survived. So, Age variable and Survived variable are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and I can say that there is no a difference between survived mean and no survived. So, Age variable and Survived variable are not associated.')

# Calculating confidence interval at 95% of confidence
tTest = stats.t.ppf(1 - alpha / 2, df = df)
ci = (diffMean - tTest*se, diffMean + tTest*se)
print('t-Test at 95%', tTest)
print('Confidence interval at 95% of confidence:', ci)

del alpha, nSurvived, nNoSurvived, varSurvived, varNoSurvived, diffMean, se, tTest, df, p_value, ci
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (12, 15))

sns.boxplot(x = dsTrain['Survived'], y = dsTrain['Fare'], ax = ax1)
ax1.set_title('Boxplot')

ax2.hist(dsTrain['Fare'], bins = 50)
ax2.set_title('Fare distribution')

ax3.scatter(dsTrain['Fare'], dsTrain['Survived'])
ax3.set_title('Fare relation')

plt.show()
dsTrain[['Fare', 'Survived']].groupby(['Survived']).agg(['mean', 'count']).reset_index()
# Calculating p-value removing outliers
alpha = 0.05 #Significance level
nSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 1)]['Fare'].count()
nNoSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 0)]['Fare'].count()
varSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 1)]['Fare'].var()
varNoSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 0)]['Fare'].var()
meanSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 1)]['Fare'].mean()
meanNoSurvived = dsTrain[(dsTrain['Fare'] <= 300) & (dsTrain['Survived'] == 0)]['Fare'].mean()
diffMean = meanSurvived - meanNoSurvived
se = np.sqrt((varSurvived / nSurvived) + (varNoSurvived / nNoSurvived))

print('Null hypothesis:',  0)
print('Alternative hypothesis:',  diffMean)
print('Standard error:', se)

tTest = (diffMean - 0) / se
print('t-Test:', tTest)

df = np.min([nSurvived - 1, nNoSurvived - 1])
print('Degrees of freedom:', df)

p_value = (1 - stats.t.cdf(tTest, df = df)) * 2 #two-sided test
print('p-value', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and I can say that there is a difference between survived mean and no survived. So, Fare variable and Survived variable are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and I can say that there is no a difference between survived mean and no survived. So, Fare variable and Survived variable are not associated.')

# Calculating confidence interval at 95% of confidence
tTest = stats.t.ppf(1 - alpha / 2, df = df)
print('t-Test at 95%', tTest)

ci = (diffMean - tTest*se, diffMean + tTest*se)
print('Confidence interval at 95% of confidence:', ci)

del alpha, nSurvived, nNoSurvived, varSurvived, varNoSurvived, diffMean, se, tTest, df, p_value, ci
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 8))

sns.countplot(dsTrain['SibSp'], hue = dsTrain['Survived'], ax = ax1)
ax1.set_title('Number of siblings and spouses frequency')
ax1.legend(['No Survived', 'Survived'], loc = 'upper right')

sns.countplot(dsTrain['Parch'], hue = dsTrain['Survived'], ax = ax2)
ax2.set_title('Number of parents and children frequency')
ax2.legend(['No Survived', 'Survived'], loc = 'upper right')

plt.show()
# Create a new variable FamilySize to make easier the analysis, and SibSp and Parch are very similar.
dsTrain['FamilySize'] = dsTrain['SibSp'] + dsTrain['Parch']
dsTrainPivot =  dsTrain[['PassengerId', 'FamilySize', 'Survived']].groupby(['FamilySize', 'Survived']).count().reset_index()
dsTrainPivot.rename(columns = { 'PassengerId': 'Count' }, inplace = True)
dsTrainPivot = dsTrainPivot.pivot(index = 'Survived', columns = 'FamilySize', values = 'Count')
dsTrainPivot.fillna(0, inplace = True)
dsTrainPivot
alpha = 0.05
chiSquareTest = 0
for r in range(dsTrainPivot.values.shape[0]):
  for c in range(dsTrainPivot.values.shape[1]):
    expectedCount = dsTrainPivot.values[r, :].sum() * dsTrainPivot.values[:, c].sum() / dsTrainPivot.values.sum()
    chiSquareTest = chiSquareTest + ((dsTrainPivot.values[r, c] - expectedCount)**2) / expectedCount
print('Chi-square test:', chiSquareTest)

df = (dsTrainPivot.shape[0] - 1) * (dsTrainPivot.shape[1] - 1)
print('Degrees of freedom', df)

p_value = 1 - stats.chi2.cdf(chiSquareTest, df = df)
print('p-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and to say that variable Survived and variable FamilySize are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and to say that variable Survived and variable FamilySize are not associated.')

del dsTrainPivot
dsTrain['Pclass'] = dsTrain['Pclass'].astype('category')
dsTrain[['Pclass', 'Name', 'Sex' ,'Ticket', 'Cabin', 'Embarked']].describe(include='all')
dsTrainPivot = dsTrain[['PassengerId', 'Pclass', 'Survived']].groupby(['Survived', 'Pclass'], as_index = False).count()
dsTrainPivot.rename(columns = { 'PassengerId': 'Count' }, inplace = True)
dsTrainPivot = dsTrainPivot.pivot(index = 'Survived', columns = 'Pclass', values = 'Count')
dsTrainPivot
alpha = 0.05
chiSquareTest = 0
for r in range(dsTrainPivot.values.shape[0]):
  for c in range(dsTrainPivot.values.shape[1]):
    expectedCount = dsTrainPivot.values[r, :].sum() * dsTrainPivot.values[:, c].sum() / dsTrainPivot.values.sum()
    chiSquareTest = chiSquareTest + ((dsTrainPivot.values[r, c] - expectedCount)**2) / expectedCount
print('Chi-square test:', chiSquareTest)

df = (dsTrainPivot.shape[0] - 1) * (dsTrainPivot.shape[1] - 1)
print('Degrees of freedom', df)

p_value = 1 - stats.chi2.cdf(chiSquareTest, df = df)
print('P-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and to say that variable survived and variable Pclass are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and to say that variable survived and variable Pclass are not associated.')

#chi2, p_value = scipy.stats.chi2_contingency(dsTrainPClass.values)[0:2]
del dsTrainPivot
dsTrainPivot = dsTrain[['PassengerId', 'Sex', 'Survived']].groupby(['Survived', 'Sex'], as_index = False).count()
dsTrainPivot.rename(columns = { 'PassengerId': 'Count' }, inplace = True)
dsTrainPivot = dsTrainPivot.pivot(index = 'Survived', columns = 'Sex', values = 'Count')
dsTrainPivot
alpha = 0.05
chiSquareTest = 0
for r in range(dsTrainPivot.values.shape[0]):
  for c in range(dsTrainPivot.values.shape[1]):
    expectedCount = dsTrainPivot.values[r, :].sum() * dsTrainPivot.values[:, c].sum() / dsTrainPivot.values.sum()
    chiSquareTest = chiSquareTest + ((dsTrainPivot.values[r, c] - expectedCount)**2) / expectedCount
print('Chi-square test:', chiSquareTest)

df = (dsTrainPivot.shape[0] - 1) * (dsTrainPivot.shape[1] - 1)
print('Degrees of freedom', df)

p_value = 1 - stats.chi2.cdf(chiSquareTest, df = df)
print('P-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and to say that variable survived and variable Sex are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and to say that variable survived and variable Sex are not associated.')

#chi2, p_value = scipy.stats.chi2_contingency(dsTrainSex.values)[0:2]
del dsTrainPivot
dsTrainPivot = dsTrain[['PassengerId', 'Embarked', 'Survived']].groupby(['Survived', 'Embarked'], as_index = False).count()
dsTrainPivot.rename(columns = { 'PassengerId': 'Count' }, inplace = True)
dsTrainPivot = dsTrainPivot.pivot(index = 'Survived', columns = 'Embarked', values = 'Count')
dsTrainPivot
alpha = 0.05
chiSquareTest = 0
for r in range(dsTrainPivot.values.shape[0]):
  for c in range(dsTrainPivot.values.shape[1]):
    expectedCount = dsTrainPivot.values[r, :].sum() * dsTrainPivot.values[:, c].sum() / dsTrainPivot.values.sum()
    chiSquareTest = chiSquareTest + ((dsTrainPivot.values[r, c] - expectedCount)**2) / expectedCount
print('Chi-square test:', chiSquareTest)

df = (dsTrainPivot.shape[0] - 1) * (dsTrainPivot.shape[1] - 1)
print('Degrees of freedom', df)

p_value = 1 - stats.chi2.cdf(chiSquareTest, df = df)
print('P-value:', p_value)

if p_value < alpha:
  print('Conclusion:', 'There is enough evidence to reject null hypothesis, and to say that variable survived and variable Embarked are associated.')
else:
  print('Conclusion:', 'There is not enough evidence to reject null hypothesis, and to say that variable survived and variable Embarked are not associated.')

#chi2, p_value = scipy.stats.chi2_contingency(dsTrainPivot.values)[0:2]
del dsTrainPivot
dsTest.head()
dsTest.shape
dsTest.info()
dsTest[['PassengerId','Age', 'SibSp', 'Parch', 'Fare']].describe()
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(12, 10))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
ax1.hist(dsTest['Age'])
ax1.set_xlabel('Age')
ax1.set_title('Age distribution')

ax2.hist(dsTest['SibSp'])
ax2.set_xlabel('SibSp')
ax2.set_title('Number of siblings / spouses distribution')

ax3.hist(dsTest['Parch'])
ax3.set_xlabel('Parch')
ax3.set_title('Number of parents / children distribution')

ax4.hist(dsTest['Fare'], bins = 50)
ax4.set_xlabel('Fare')
ax4.set_title('Fare distribution')

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
sns.boxplot(dsTest['Age'], ax = ax1)
ax1.set_xlabel('Age')
ax1.set_title('Age distribution')

sns.boxplot(dsTest['SibSp'], ax = ax2)
ax2.set_xlabel('SibSp')
ax2.set_title('Number of siblings / spouses distribution')

sns.boxplot(dsTest['Parch'], ax = ax3)
ax3.set_xlabel('Parch')
ax3.set_title('Number of parents / children distribution')

sns.boxplot(dsTest['Fare'], ax = ax4)
ax4.set_xlabel('Fare')
ax4.set_title('Fare distribution')

plt.show()
dsTest['Pclass'].value_counts()
dsTest['Sex'].value_counts()
dsTest['Embarked'].value_counts()
dsTrain.info()
dsTrain['Title'] = dsTrain['Name'].map(lambda x: re.findall('([A-Za-z]+)\.', x)[0])
dsTrain['FamilySize'] = dsTrain['SibSp'] + dsTrain['Parch']
dsTrain['Age'] = dsTrain['Age'].fillna(dsTrain.groupby('Title')['Age'].transform('median'))
dsTrain['Age'].isna().sum()
dsTrain['Embarked'] = dsTrain['Embarked'].fillna('C')
dsTrain['Embarked'].isna().sum()
def setClassAge(x):
  '''
    Objective: categorize age by range.
    Inputs:
      x (number): age
  '''
  if x <= 15:
    return 0 # Kid
  elif x > 15 and x <= 30:
    return 1 # Young
  elif x > 30 and x <= 50:
    return 2 # Adult
  else:
    return 3 # Mayor
dsTrain['Age'] = dsTrain['Age'].astype('int')
dsTrain['AgeClass'] = dsTrain['Age'].apply(setClassAge)
def setClassFare(x):
  '''
    Objective: categorize fare by range
    Inputs:
      x (number): fare
  '''
  if x <= 50:
    return 0
  elif x > 50 and x <= 100:
    return 1
  elif x > 100 and x <= 150:
    return 2
  elif x > 150 and x <= 200:
    return 3
  elif x > 200 and x <= 250:
    return 4
  elif x > 250 and x <= 300:
    return 5
  elif x > 300 and x <= 350:
    return 6
  elif x > 400 and x <= 450:
    return 7
  elif x > 450 and x <= 500:
    return 8
  else:
    return 9
dsTrain['FareClass'] = dsTrain['Fare'].apply(setClassFare)
dsTrain['FamilySize'] = dsTrain['FamilySize'].astype('category')
dsTrain['Survived'] = dsTrain['Survived'].astype('category')
dsTrain['Pclass'] = dsTrain['Pclass'].astype('category')
dsTrain['AgeClass'] = dsTrain['AgeClass'].astype('category')
dsTrain['FareClass'] = dsTrain['FareClass'].astype('category')
dsTrain.head()
dsTrain.drop(['PassengerId', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'Fare'], axis = 1, inplace = True)
dsTrain.head()
x_train = dsTrain.drop('Survived', axis = 1)
y_train = dsTrain['Survived']
x_train = pd.get_dummies(x_train, drop_first = True)
x_train.head()
x_train.shape
dsTest.info()
dsTest['Title'] = dsTest['Name'].map(lambda x: re.findall('([A-Za-z]+)\.', x)[0])
dsTest['FamilySize'] = dsTest['SibSp'] + dsTest['Parch']
dsTest['Age'] = dsTest['Age'].fillna(dsTest.groupby('Title')['Age'].transform('median'))
dsTest['Age'].isna().sum()
dsTest['Age'] = dsTest['Age'].fillna(dsTest['Age'].median())
dsTest['Age'].isna().sum()
dsTest['Embarked'] = dsTest['Embarked'].fillna('C')
dsTest['Embarked'].isna().sum()
dsTest['Age'] = dsTest['Age'].astype('int')
dsTest['AgeClass'] = dsTest['Age'].apply(setClassAge)
dsTest['FareClass'] = dsTest['Fare'].apply(setClassFare)
dsTest['FamilySize'] = dsTest['FamilySize'].astype('category')
dsTest['Pclass'] = dsTest['Pclass'].astype('category')
dsTest['AgeClass'] = dsTest['AgeClass'].astype('category')
dsTest['FareClass'] = dsTest['FareClass'].astype('category')
dsTest.head()
dsTest.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Title', 'Fare'], axis = 1, inplace = True)
dsTest.head()
# PassengerId
x_test = pd.get_dummies(dsTest[['Pclass', 'Sex', 'Embarked', 'FamilySize', 'AgeClass', 'FareClass']], drop_first = True)
x_test.head()
x_test = pd.concat([dsTest['PassengerId'], x_test], axis = 1)
print(x_test.shape)
x_test.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Val set:', X_val.shape,  y_val.shape)
X_test = x_test.drop('PassengerId', axis = 1).values
print('Test set:', X_test.shape)
k = 15
mean_acc = np.zeros((k - 1))
std_acc = np.zeros((k - 1))
confusion_matrices = []
f1s = []

for n in range(1, k):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train, y_train)
    yhat = knn.predict(X_val)

    mean_acc[n-1] = metrics.accuracy_score(y_val, yhat)
    std_acc[n-1] = np.std(yhat == y_val) / np.sqrt(yhat.shape[0])
    confusion_matrices.append(metrics.confusion_matrix(y_val, yhat, labels = [0, 1]))
    f1s.append(metrics.f1_score(y_val, yhat))

mean_acc
plt.plot(range(1, k), mean_acc, 'g')
plt.fill_between(range(1, k), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Val accuracy ', '+/- 3xstd'))
plt.ylabel('Val accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
# what percentage of correct predictions were made?
print("The best accuracy was ", mean_acc.max(), "with k=", mean_acc.argmax() + 1) 
print('F1-score:', f1s[mean_acc.argmax()])
irs = np.linspace(0.01, 1, num = 10)
mean_acc = np.zeros(len(irs))
confusion_matricesLR = []
f1s = []
       
for i in range(len(irs)):
  LR = LogisticRegression(C = irs[i], solver = 'liblinear').fit(X_train, y_train)
  yhat = LR.predict(X_val)
  yhat_prob = LR.predict_proba(X_val)
  mean_acc[i] = metrics.accuracy_score(y_val, yhat)
  #confusion_matricesLR.append(metrics.confusion_matrix(y_val, yhat, labels = [0, 1]))
  f1s.append(metrics.f1_score(y_val, yhat))

mean_acc
plt.plot(irs, mean_acc, 'g')
plt.ylabel('Val accuracy ')
plt.xlabel('Inverse of regularization strength')
plt.tight_layout()
plt.show()
print("The best accuracy was ", mean_acc.max(), "with C = ", irs[mean_acc.argmax()])
print('F1-score:', f1s[mean_acc.argmax()])
x_test
knn = KNeighborsClassifier(n_neighbors = 10).fit(X_train, y_train)
y_predTest = knn.predict(X_test)
y_predTest
dsSubmission = pd.DataFrame({'PassengerId': x_test['PassengerId'].values, 'Survived': y_predTest})
dsSubmission.head()
dsSubmission.to_csv('titanic_submission.csv', index=False)
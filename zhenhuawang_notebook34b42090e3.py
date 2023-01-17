import numpy as np

import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

import matplotlib.pyplot as plt
# read files

df = pd.read_csv('../input/train.csv')

dftest = pd.read_csv('../input/test.csv')



# show datas

df.head(2)

df.shape
# show data structure

df.info()
%matplotlib inline



plt.title('Survived(1) or not(0)')

df.Survived.value_counts().plot(kind='bar')
plt.title('Pclass distribution (1=1st, 2=2nd, 3=3rd)')

df.Pclass.value_counts().plot(kind='bar')
plt.title('Pclass and Survial')

plt.hold

df.Pclass.value_counts().plot(kind='bar')

df.Pclass[df.Survived==1].value_counts().plot(kind='bar', colormap='cubehelix')
plt.title('Survial and age')

plt.hold

df.Age.plot(kind='hist')

df.Age[df.Survived==1].plot(kind='hist')

plt.title('Sex and Survial')

plt.hold

df.Sex[df.Survived==1].value_counts().plot(kind='bar', colormap='winter')
df = df.drop(['PassengerId','Name','Ticket', 'Cabin','Embarked'], axis=1)

dftest = dftest.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

df.info()
avg_age = df['Age'].mean()

std_age = df['Age'].std()

sum_nan_age = df['Age'].isnull().sum()



# generate random integers (ages)

rand_age = np.random.randint(avg_age-std_age, avg_age+std_age, sum_nan_age)



# fill NaN with generated random integers

index_nan = df['Age'].isnull()

df['Age'][index_nan] = rand_age





# the same for test

avg_age = dftest['Age'].mean()

std_age = dftest['Age'].std()

sum_nan_age = dftest['Age'].isnull().sum()

rand_age = np.random.randint(avg_age-std_age, avg_age+std_age, sum_nan_age)

index_nan = dftest['Age'].isnull()

dftest['Age'][index_nan] = rand_age
df['Sex'].isnull().sum()
dftest['Sex'].isnull().sum()
df = pd.get_dummies(df, columns=['Sex'])

df.drop(['Sex_male'], axis=1, inplace=True)



dftest = pd.get_dummies(dftest, columns=['Sex'])

dftest.drop(['Sex_male'], axis=1, inplace=True)
df['Pclass'].isnull().sum()
dftest['Pclass'].isnull().sum()
df = pd.get_dummies(df, columns=['Pclass'])

df.drop(['Pclass_3'], axis=1, inplace=True)



dftest = pd.get_dummies(dftest, columns=['Pclass'])

dftest.drop(['Pclass_3'], axis=1, inplace=True)
df['Fare'].isnull().sum()
dftest['Fare'].isnull().sum()
avg_fare = dftest['Fare'].mean()

std_fare = dftest['Fare'].std()

sum_nan_fare = dftest['Fare'].isnull().sum()



# generate random integers (ages)

rand_fare = np.random.randint(avg_fare-std_fare, avg_fare+std_fare, sum_nan_fare)



# fill NaN with generated random integers

index_nan = dftest['Fare'].isnull()

dftest['Fare'][index_nan] = rand_fare
df.info()
# all data

x = df.drop(['Survived'], axis=1)

y = df['Survived']



# training data

train_num = 800

x_train = x.iloc[train_num:,:]

y_train = y.iloc[train_num:]



# dev data

x_dev = x.iloc[:train_num,:]

y_dev = y.iloc[:train_num]



# test data

x_test = dftest.drop(['PassengerId'], axis=1)



x_train.shape


svm_clf = Pipeline([

    ('Scalar', StandardScaler()),

    ('svc', SVC(kernel='linear', gamma=0.5, C=4))

])

svm_clf.fit(x_train, y_train)



# compare test and dev score: hyperparameter tuning

print(svm_clf.score(x, y))

print(svm_clf.score(x_dev, y_dev))



# 0.5, 4
# prediction

pred = svm_clf.predict(x_test)



# save to file

df_pred = pd.DataFrame({'PassengerId': dftest['PassengerId'],

                        'Survived': pred })

df_pred.to_csv('titanic.csv', index=False)
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler 
train_set = pd.read_csv('../input/train.csv')
feature1 = 'Age'

feature2 = 'Fare'
train_set = train_set.dropna(subset=['Age','Fare','Survived'], how='any')
train_x = train_set[[feature1,feature2]]

train_y = train_set.Survived
scaler = StandardScaler()

scaler.fit(train_x)
logReg = LogisticRegression()

logReg.fit(scaler.transform(train_x),train_y)
test_set = pd.read_csv('../input/test.csv')

print(test_set.count())

result_set = pd.read_csv('../input/gender_submission.csv')

test_set = test_set.merge(result_set)
test_set = test_set.interpolate()
test_x = test_set[['Age','Fare']]

test_y = test_set.Survived
logReg.score(scaler.transform(test_x),test_y)
predict = logReg.predict(scaler.transform(test_x))

test = test_x.values.tolist()



for i in range(len(predict)):

    if(predict[i] == 0):

        plt.plot(test[i][0],test[i][1],'ro')

    else:

        plt.plot(test[i][0],test[i][1],'bo')

plt.ylabel('Fare')

plt.xlabel('Age')

plt.title('Blue: Will Survive | Red: Won\'t Survive')

plt.show()
import csv



with open('titanic_my_predictions.csv', mode='w') as csv_file:

    fieldnames = ['PassengerId', 'Survived']

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)



    writer.writeheader()

    id_list = test_set.PassengerId.tolist()

    for i in range(len(predict)):

        writer.writerow({'PassengerId': id_list[i], 'Survived': predict[i]})
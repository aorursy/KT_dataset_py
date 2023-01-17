# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# ability to build simple plots
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dta_1 = pd.read_csv('../input/train.csv')
dta_1.describe()
dta_1['Fare'].describe()
survived = dta_1.query('Survived > 0')
survived['Fare'].describe()
died = dta_1.query('Survived == 0')
died['Fare'].describe()
from sklearn import tree
input_dta = dta_1.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
input_dta['Embarked'] = input_dta['Embarked'].map({ 'S':0, 'C':1, 'Q':2})
input_dta['Sex'] = input_dta['Sex'].map({ 'male': 0, 'female': 1})

output_dta = dta_1['Survived']

# fill NaN in test_input
input_dta = input_dta.fillna(0)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(input_dta, output_dta)
test_sample = dta_1.sample(n=20)

test_input = test_sample.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
test_input['Embarked'] = test_input['Embarked'] = test_input['Embarked'].map({ 'S':0, 'C':1, 'Q':2})
test_input['Sex'] = test_input['Sex'].map({ 'male': 0, 'female': 1})
test_output = test_sample['Survived']

# fill NaN in test_input
test_input = test_input.fillna(0)

test_sample

test_result = pd.DataFrame([clf.predict(test_input), test_sample['Survived']]).T
test_result.columns = ['Tested', 'Expected']
test_result.T

spec = pd.read_csv('../input/test.csv')
spec = spec.drop(['Name', 'Ticket', 'Cabin'], axis=1)
spec['Embarked'] = spec['Embarked'].map({ 'S':0, 'C':1, 'Q':2})
spec['Sex'] = spec['Sex'].map({ 'male': 0, 'female': 1})
spec = spec.fillna(0)

res = clf.predict(spec)

spec['Survived'] = res
result = spec[['PassengerId', 'Survived']]
result.set_index('PassengerId', inplace=True)
result.to_csv('results.csv', header=True)
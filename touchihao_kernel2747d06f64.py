# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test_id = test["PassengerId"]

drop_col = data.columns[np.unique(np.where(data.isnull())[1])]

del data[drop_col[0]], data[drop_col[1]], data[drop_col[2]]

del test[drop_col[0]], test[drop_col[1]], test[drop_col[2]]







label = data[data.columns[1]]

drop_col2 = data.columns[[0,1,-2]]

del data[drop_col2[0]], data[drop_col2[1]], data[drop_col2[2]], data['Name'], data['Fare']

del test[drop_col2[0]], test[drop_col2[2]], test['Name'], test['Fare']



data.shape,label.shape,test.shape
man_survived = sum(label.iloc[np.where(data["Sex"]=="male")]==1)

man_nonsurvived = sum(label.iloc[np.where(data["Sex"]=="male")]==0)



woman_survived = sum(label.iloc[np.where(data["Sex"]=="female")]==1)

woman_nonsurvived = sum(label.iloc[np.where(data["Sex"]=="female")]==0)



survived = [man_survived, woman_survived]

nonsurvived = [man_nonsurvived, woman_nonsurvived]

print(man_survived, man_nonsurvived, woman_survived, woman_nonsurvived)
plt.bar([0,1],survived,color='salmon',edgecolor='white')

plt.bar([0,1],nonsurvived, bottom=survived,color='gray',edgecolor='white')



plt.xticks([0,1], ['male', 'female'])

plt.xlabel("Sex")

plt.ylabel("person")



plt.show()
def str_to_num(data):

    for i in range(len(data)):

        if data['Sex'][i]=='male':

            data['Sex'][i]=1

        else:

            data['Sex'][i]=0

    return data
data = str_to_num(data)

test = str_to_num(test)
RF_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

RF_model.fit(data,label)

predictions = RF_model.predict(test)
output = pd.DataFrame({'PassengerId': test_id,'Survived': predictions})

output.to_csv('my_submission.csv', index= False)
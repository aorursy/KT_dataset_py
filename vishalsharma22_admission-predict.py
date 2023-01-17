# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

data.head()
data.hist(figsize = (10,10))

plt.show()
import seaborn as sns

corr = data.corr()

f,ax=plt.subplots(figsize=(20,1))

sns.heatmap(corr.sort_values(by=['Chance of Admit '],ascending=False).head(1), cmap='Blues')

plt.title("features correlation with the Research", weight='bold', fontsize=18)

plt.xticks(weight='bold')

plt.yticks(weight='bold', color='dodgerblue', rotation=0)

plt.show()
real_x=data.iloc[:,1:-1].values

real_y=data.iloc[:,-1].values
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=.30,random_state=0)
reg=RandomForestRegressor( n_estimators=300)

reg.fit(train_x,train_y)

pred_y=reg.predict(test_x)
test_y
pred_y
x=[337,118,4,4.5,4.5,9.65,1]

pred_y=reg.predict([x])

pred_y
print(reg.score(train_x,train_y)*100)

print(reg.score(test_x,test_y)*100)
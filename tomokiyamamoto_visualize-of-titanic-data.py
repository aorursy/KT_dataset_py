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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
titanic_gendersub_data = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

titanic_test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

titanic_train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_train_data.head()
sns.scatterplot(x=titanic_train_data['Age'],y=titanic_train_data['Pclass'])
sns.scatterplot(x=titanic_train_data['Age'],y=titanic_train_data['Fare'])
sns.regplot(x=titanic_train_data['Age'],y=titanic_train_data['Fare'])
sns.scatterplot(x=titanic_train_data['Age'],y=titanic_train_data['Fare'],hue=titanic_train_data['Survived'])
plt.figure(figsize=(8,16))

sns.lmplot(x="Age",y="Fare",hue="Survived",data=titanic_train_data)
sns.swarmplot(x=titanic_train_data['Survived'],y=titanic_train_data['Fare'])
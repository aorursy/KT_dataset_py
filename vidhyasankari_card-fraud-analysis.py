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
#getting helpers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data.head()

data.describe()
data.info()
_=[x for x in data['Class']==0]
__=[y for y in data['Class']==1]
len(_),len(__)
y=data['Class']
x=data.drop('Class',axis=1)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
model.fit(xTrain,yTrain)
yPred=model.predict(xTest)
accuracy = metrics.accuracy_score(yTest, yPred)
accuracy_percentage = 100 * accuracy
accuracy_percentage
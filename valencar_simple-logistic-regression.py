# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#import

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
df = pd.read_csv("../input/creditcard.csv",header = 0)

#print type(df)

df.head()
df.shape
df['Class'].value_counts()
#split into test train

traindf, testdf = train_test_split(df, test_size = 0.4)
#divide into X and Y

XCol = df.columns[0:30]

YCol = "Class"  
logReg = LogisticRegression()

logReg.fit(traindf[XCol],traindf[YCol])
predictions = logReg.predict(testdf[XCol]);

accuracy = metrics.accuracy_score(predictions,testdf[YCol])

accuracy
# Get the sum of amounts per Class

df1 = df.groupby('Class')['Amount'].sum()

df1
# Get the mean of amounts per Class

# Class: 0 - Normal, 1 - Fraud

df1 = df.groupby('Class')['Amount'].mean()

df1
import seaborn as sea



sea.barplot(x='Class',y='Amount', data=df)
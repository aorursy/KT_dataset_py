# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filename = '../input/pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

df=pd.read_csv("../input/"+filename,names=names)

skew=df.skew()
print(skew)

#for normally distributed data skewness should be close to zero. If skewness>0 then

#it means that there is more weight on the left tail of the distribution and if skewness<0 then it means that

# there is more weight on the right tail of the distribution
df.head()
df.hist(figsize=(10,8))

plt.show()
corr=df.corr()

sns.heatmap(corr,vmax=0.8)
#lets see the relationship between our independent variables and dependent variable(Class)

sns.barplot("class","preg",data=df)
sns.barplot("class","plas",data=df)
sns.barplot("class","pres",data=df)
sns.barplot("class","skin",data=df)
sns.barplot("class","test",data=df)
sns.barplot("class","mass",data=df)
sns.barplot("class","pedi",data=df)
sns.barplot("class","age",data=df)
X=df.iloc[:,0:8]

y=df.iloc[:,8]

X.head()

#Spliiting our dataset into training and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Applying Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Applying logistic Regression

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
#Calculating Y_predection

y_pred=classifier.predict(X_test)
#Forming a confusion matrix to check our accuracy

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
acc=(98+29)/(98+29+18+9)

print(acc)

#We got an accuracy of 82.46%
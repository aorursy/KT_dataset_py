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
import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/HR_comma_sep.csv")
#Let see first five observations of our data

df.head()

df.isnull().sum()

#Our data dont have any missing values :D
#lets check the data types of our variiables

df.dtypes

dfcorr=df.drop(["Work_accident","sales","promotion_last_5years","salary","left"],axis=1)

corr=dfcorr.corr()

plt.figure(figsize=(5,5))

sns.heatmap(corr,vmax=0.6,annot=True)

#Heat map shows correlations among dif
sns.pairplot(df)
count=df.groupby(df["sales"]).count()

count = pd.DataFrame(count.to_records())

count = count.sort_values(by= 'left', ascending = False)

count = count['sales']
sns.countplot(y='sales', data=df, order=count)
count=df.groupby(df["salary"]).count()

count=pd.DataFrame(count.to_records())

count=count["salary"]

count
sns.countplot(x="salary",data=df)
sns.barplot("time_spend_company","left",data=df)

#employess who have worked for 5 years in a company are more likely to leave
sns.barplot("Work_accident","left",data=df)

#Employees who did not meet an accident are more likely to leave o.O
sns.barplot("promotion_last_5years","left",data=df)

# Employees who did bot get promotion in the last five years are more likely to leave and it makes sense. 
sns.barplot("salary","left",data=df)

#Employees having low salaries are more likely to leave
sns.barplot("number_project","left",data=df)

#Employees having the most and least project are most likely to leave
sns.barplot("sales","left",data=df)

#Management and R&D are more likely to stay and HR and Accounting are more likely to leave
sns.barplot(df["last_evaluation"],df["left"])

#Employees who were evaluated long time ago or the empolyees who are evaluated recently are more likely to leave


X=df.iloc[:,0:10]

y=df.iloc[:,6]

X.drop("left",inplace=True,axis=1)

#Splitting the dataset into trainiing and test set with 70% of the data into the training set 

#and 30% data on the test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#Creating dummy variables for categorical variables

X_train=pd.get_dummies(X_train)

X_test=pd.get_dummies(X_test)
#Applying feature scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)
#Applying Random forest

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=100,random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#Creating confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
accuracy=(3450+1001)/(3450+1001+37+12)

print(accuracy)

#We get an accuracy of 98.91% :D
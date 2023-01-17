import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas_profiling
df = pd.read_csv('../input/titanic/train.csv') # training data set
df.head(12)  # Sibsp - Number of Siblings/Spouses Aboard. Parch - Number of Parents/Children Aboard
df.info() #here age and cabin have null values, so we need to see how to manage that by either dropping the columns or substitutingthe nan values
#report = pandas_profiling.ProfileReport(df) # to understand intial data and to reduce further work in eda
#report
#sns.pairplot(df) # to understand initial data
corr = df.corr() #for finding correlation
corr
df.describe() #to understand the data better
sns.heatmap(corr,annot=True) # for easier understanding
sns.countplot("Survived", data = df)
plt.hist(df.Age)
sns.countplot("Survived",hue = "Sex", data = df)# female survived 3.5 times more than men
sns.countplot("Survived",hue = "Pclass", data = df) # cheapest priced class survived less
df["Fare"].plot.hist()
df.isnull().sum()
sns.heatmap(df.isnull(), yticklabels= False, cmap='YlOrBr_r') # to see visualization
sns.boxplot(x = 'Pclass', y = 'Age', data= df) # whiskers are median (+ or -) 1.5*IQR and outliers are points which doesn't fall under 1.5*IQR
df["Pclass"].plot.hist()
# Imputation - substituting or removing the NAN value columns
df.drop("Cabin",axis = 1,inplace= True) # Axis 0 represents rows, axis 1 represents columns and when inplace is True then df will permanently drop that column
df.head()
df['Age'] = df['Age'].fillna(df.Age.median())
df.dropna(subset=['Embarked'], inplace = True) # dropping the nan value columns, if we want the nan values then we can replace with mean or median based on variance or outliers
df.isnull().sum() # check if there are any nan values
df.info()
plt.hist(df.Age) # we can see the difference in histograms as we add mean value inplace of nan
#df['norm_Fare'] = np.log(df.Fare+1) #we can normalize the fare data if we like 
sex = pd.get_dummies(df['Sex'],drop_first = True)
sex.head()
Pcl = pd.get_dummies(df['Pclass'],drop_first = True )
Pcl.head()
df = pd.concat([df,sex,Pcl],axis=1)
df.head()
df.drop(['PassengerId','Name','Ticket','Sex','Embarked','Pclass'],axis = 1, inplace = True)
df.head()
x = df.drop("Survived",axis = 1)          # Independent variables
y = df['Survived']                        #for predicting dependent variable
from sklearn.model_selection import train_test_split # if both train and test data sets are in same file we can change by using this
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size = 0.5) #0.5 means 50% data for train and 50% for test
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,Y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report #for knowing the precision or we can use confusion matrix
classification_report(Y_test,prediction)
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,prediction)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,prediction)          #to know the accuracy 
test = pd.read_csv('../input/titanic/test.csv')
test.head()
test.drop("Cabin",axis = 1,inplace= True) 
test['Age'] = test['Age'].fillna(test.Age.median())
test['Fare'] = test['Fare'].fillna(test.Age.median())
sex1 = pd.get_dummies(test['Sex'],drop_first = True)
sex1.head()
id = test['PassengerId']
id.head()
Pcl1 = pd.get_dummies(test['Pclass'],drop_first = True )
Pcl1.head()
test = pd.concat([test,sex1,Pcl1],axis=1)
test.head()
test.drop(['PassengerId','Name','Ticket','Sex','Embarked','Pclass'],axis = 1, inplace = True)
test.isnull().sum()        # check if there are any nan values
test.head()
predictions = logmodel.predict(test)        
submission = pd.DataFrame({'PassengerId':id,'Survived':predictions})  
submission.head()                                                     
# convert to csv file
os.chdir("/kaggle/working/")

filename = 'submission.csv'

submission.to_csv(filename,index=False) 
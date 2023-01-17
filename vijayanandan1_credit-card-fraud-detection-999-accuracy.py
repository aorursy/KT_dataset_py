import pandas as pd

data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
data.columns
import matplotlib.pyplot as plt

plt.hist(data.Class)
data.isnull().sum()
data.info()

data.describe()

a = data[data['Class']==1]

len(a)
len(data["Class"]) - len(a)
data.hist(figsize=(20,20),color='blue')

plt.show()
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
X=data.drop(['Class'],axis=1)

y=data['Class']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=123)

dt=DecisionTreeRegressor()

model1=dt.fit(X_train,y_train)

prediction1=model1.predict(X_test)

DT = accuracy_score(y_test,prediction1)

DT
from sklearn import neighbors

classifierkn=neighbors.KNeighborsClassifier()
X2=data.drop(['Class'],axis=1)

y2=data['Class']

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.30,random_state=123)

#Training the Model Using KN

model2 = classifierkn.fit(X2_train,y2_train)

prediction2=model2.predict(X2_test)

KN = accuracy_score(y_test,prediction2)

KN
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
X3=data.drop(['Class'],axis=1)

y3=data['Class']

X3_train,X3_test,y3_train,y3_test=train_test_split(X3,y3,test_size=0.30,random_state=123)

#Training the Model Using RF

model3 = rfc.fit(X3_train,y3_train)

prediction3=model3.predict(X3_test)

RF = accuracy_score(y_test,prediction3)

RF
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
X4=data.drop(['Class'],axis=1)

y4=data['Class']

X4_train,X4_test,y4_train,y4_test=train_test_split(X4,y4,test_size=0.30,random_state=123)

#Training the Model Using NB

model4 = gnb.fit(X4_train,y4_train)

prediction4=model4.predict(X4_test)

NB = accuracy_score(y_test,prediction4)

NB
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
X5=data.drop(['Class'],axis=1)

y5=data['Class']

X5_train,X5_test,y5_train,y5_test=train_test_split(X5,y5,test_size=0.30,random_state=123)

#Training the Model Using LR

model5 = lr.fit(X5_train,y5_train)

prediction5=model5.predict(X5_test)

LR = accuracy_score(y_test,prediction5)

LR
from sklearn import svm

clf = svm.SVC()
X6=data.drop(['Class'],axis=1)

y6=data['Class']

X6_train,X6_test,y6_train,y6_test=train_test_split(X6,y6,test_size=0.30,random_state=123)

#Training the Model Using SVM

model6 = clf.fit(X6_train,y6_train)

prediction6=model6.predict(X6_test)

svm = accuracy_score(y_test,prediction6)

svm
data = [['Decision Tree',DT ],['KNN',KN ],['Random Forest',RF],['Naive Bayes',NB] , ['Logistic Regression',LR],['SVM',svm]]

df = pd.DataFrame(data,columns=['Algorithms','Accuracy'])

df
from pandas import DataFrame

import matplotlib.pyplot as plt

Data = {'Algorithms': ['Decision Tree','KNN','Random Forest','Naive Bayes','Logistic Regression' , 'SVM'],

        'Accuracy': [DT,KN,RF,NB,LR,svm]

       }

df = DataFrame(Data,columns=['Algorithms','Accuracy'])

df.plot(x ='Algorithms', y='Accuracy', kind = 'bar')



y_pos = [0,1,5,8,9]

 

plt.show()
df.describe()
import numpy as np

import pandas as pd

import os, sys

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score





from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.svm import SVC



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.linear_model import RidgeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import  LinearSVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier
df=pd.read_csv("../input/parkinsons-data-set/parkinsons.data")

df.head()
df.shape
df.dtypes
df.describe()
df.info()
df.isnull().sum()
print("maximum value=",df['MDVP:Fo(Hz)'].max())

print("maximum value=",df['MDVP:Fhi(Hz)'].max())

print("maximum value=",df['MDVP:Flo(Hz)'].max())

print("maximum value=",df['MDVP:Jitter(%)'].max())

print("maximum value=",df['PPE'].max())
print("manimum value=",df['MDVP:Fo(Hz)'].min())

print("manimum value=",df['MDVP:Fhi(Hz)'].min())

print("manimum value=",df['MDVP:Flo(Hz)'].min())

print("manimum value=",df['MDVP:Jitter(%)'].min())

print("manimum value=",df['PPE'].min())
sns.catplot(x='status',kind='count',data=df)
x=df['MDVP:Fo(Hz)']

y=df['MDVP:Flo(Hz)']

N = 195

colors = np.random.rand(N)

area = (25 * np.random.rand(N))**2

df1 = pd.DataFrame({'X': x,'Y': y,'Colors': colors,"bubble_size":area})
plt.scatter('X', 'Y',  s='bubble_size', alpha=0.5, data=df1)

plt.xlabel("X", size=16)

plt.ylabel("y", size=16)

plt.title("Bubble Plot with Matplotlib", size=18)
box1=sns.catplot(x='status',y='MDVP:Jitter(Abs)',kind='box',data=df)

plt.show ()

box1=sns.catplot(x='status',y='MDVP:Jitter(Abs)',kind='box',data=df)

plt.show ()

box1=sns.catplot(y="MDVP:Flo(Hz)", x="status", data=df,kind='box')

plt.show ()

box1=sns.catplot(y="MDVP:Jitter(%)", x="status", data=df,kind='box')

plt.show ()

box1=sns.catplot(y="MDVP:Jitter(Abs)", x="status", data=df,kind='box')

plt.show ()

box1=sns.catplot(y="MDVP:RAP", x="status", data=df,kind='box')

plt.show ()
# Plot histograms for each variable

df.hist(figsize=(20,12))

plt.show()
# Create scatter plot matrix

from pandas.plotting import scatter_matrix

scatter_matrix(df, figsize = (20,20),color='m')

plt.show()
sns.pairplot(df,hue = 'status', vars = ['MDVP:Flo(Hz)','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ'] )
f,ax=plt.subplots(1,1,figsize=(25,4))

sns.kdeplot(df.loc[(df['status']==1), 'RPDE'], color='r', shade=True, Label='1')

sns.kdeplot(df.loc[(df['status']==0), 'RPDE'], color='g', shade=True, Label='0')

plt.xlabel('RPDE') 
#Heat map for correlation matrix

corrmat = df.corr()

fig = plt.figure(figsize = (45, 10))

sns.heatmap(corrmat, vmax = 0.8, square = True,annot=True)
f,axes=plt.subplots (1,1,figsize=(15,4))

sns.distplot(df['MDVP:Fo(Hz)'],kde=True,hist=True,color="r")
f,ax=plt.subplots(1,2,figsize=(20,5))

box1=sns.violinplot(x="status",y="MDVP:RAP",data=df,ax=ax[0])

box2=sns.violinplot(x="status",y="MDVP:Jitter(%)",data=df,ax=ax[1])
x=df.drop(['status','name'],axis=1)

y=df['status']
#Scaling

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(x)
#Dividing into Test and Train 

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
models = []

models.append(("LR",LogisticRegression()))

models.append(("GNB",GaussianNB()))

models.append(("KNN",KNeighborsClassifier()))

models.append(("XGB",XGBClassifier()))

models.append(("DecisionTree",DecisionTreeClassifier()))

models.append(("LDA",  LinearDiscriminantAnalysis()))

models.append(("QDA",  QuadraticDiscriminantAnalysis()))

models.append(("AdaBoost", AdaBoostClassifier()))

models.append(("SVM Linear",SVC(kernel="linear")))

models.append(("SVM RBF",SVC(kernel="rbf")))

models.append(("Random Forest",  RandomForestClassifier()))

models.append(("Bagging",BaggingClassifier()))

models.append(("Calibrated",CalibratedClassifierCV()))

models.append(("GradientBoosting",GradientBoostingClassifier()))

models.append(("LinearSVC",LinearSVC()))

models.append(("Ridge",RidgeClassifier()))
results = []

for name,model in models:

    kfold = KFold(n_splits=10, random_state=0)

    cv_result = cross_val_score(model,x_train,y_train, cv = kfold,scoring = "accuracy")

# It gives you an unbiased estimate of the actual performance you will get at runtime

    results.append(tuple([name,cv_result.mean(), cv_result.std()]))

    results.sort(key=lambda x: x[1], reverse = True)    

for i in range(len(results)):

    print('{:20s} {:2.2f} (+/-) {:2.2f} '.format(results[i][0] , results[i][1] * 100, results[i][2] * 100))
from sklearn.metrics import classification_report, confusion_matrix

xgb= XGBClassifier()

xgb.fit(x_train,y_train)

y_pred=xgb.predict(x_test)

print(classification_report(y_test,y_pred))

accuracy1=xgb.score(x_test,y_test)

print (accuracy1*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
from sklearn.ensemble import RandomForestClassifier 

des_class=DecisionTreeClassifier()

des_class.fit(x_train,y_train)

des_predict=des_class.predict(x_test)

print(classification_report(y_test,des_predict))

accuracy3=des_class.score(x_test,y_test)

print(accuracy3*100,'%')

cm = confusion_matrix(y_test, des_predict)

sns.heatmap(cm, annot= True)
from sklearn.naive_bayes  import GaussianNB 

from sklearn.metrics import classification_report, confusion_matrix

nvclassifier = GaussianNB ()

nvclassifier .fit(x_train,y_train)

y_pred=nvclassifier .predict(x_test)

print(classification_report(y_test,y_pred))

print(accuracy_score(y_pred,y_test)*100,'%')

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot= True)
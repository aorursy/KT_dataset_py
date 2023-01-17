import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import cufflinks as cf

from plotly import __version__

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()

%matplotlib inline
print( __version__)
dataset = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

dataset.head()
sns.heatmap(dataset.isnull())
dataset.describe()
dataset['DEATH_EVENT'].value_counts()
X1 = dataset.iloc[:,:-1]

y1 = dataset.iloc[:,-1]
#feature selction

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

fit_best_features = SelectKBest(score_func=chi2,k=10)

best_features=fit_best_features.fit(X1,y1)



dataset_scores = pd.DataFrame(best_features.scores_)

dataset_cols = pd.DataFrame(X1.columns)
featurescores = pd.concat([dataset_cols,dataset_scores],axis=1)

featurescores.columns=['column','scores']
featurescores
print(featurescores.nlargest(10,'scores'))
from sklearn.ensemble import ExtraTreesClassifier

ee = ExtraTreesClassifier()

ee.fit(X1,y1)
fea_imp=pd.Series(ee.feature_importances_,index=X1.columns)

fea_imp.nlargest(10).plot(kind='barh')

X2 = dataset.iloc[:,:-1]

y2 = dataset.iloc[:,-1]
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X2=ss.fit_transform(X2)

from sklearn.feature_selection import VarianceThreshold

feature_high_variance = VarianceThreshold(threshold=(0.8*(1-0.8)))

falls=feature_high_variance.fit(X2)
dataset_scores1 = pd.DataFrame(falls.variances_)

dat1 = pd.DataFrame(X1.columns)

high_variance = pd.concat([dataset_scores1,dat1],axis=1)

high_variance.columns=['variance','cols']
high_variance
high_variance[high_variance['variance']>0.8]


sns.distplot(dataset['age'],bins=30)
sns.boxplot(x='DEATH_EVENT',y='age',data=dataset)
ds = dataset['DEATH_EVENT'].value_counts().reset_index()

ds.columns = ['DEATH_EVENT', 'count']

fig = px.pie(

    ds, 

    values='count', 

    names='DEATH_EVENT',

    title='DEATH_EVENT bar chart', 

    width=500, 

    height=500

)

fig.show()
dataset.iplot(kind='bar',x='DEATH_EVENT',y='platelets')

#dataset.count().iplot(kind='bar')
dataset.iplot(kind='bar',x='DEATH_EVENT',y='time')

#sns.barplot(x='diabetes',y='DEATH_EVENT',data=dataset)
sns.violinplot(x='DEATH_EVENT',y='serum_creatinine',data=dataset)
dataset.iplot(kind='scatter',x='DEATH_EVENT',y='ejection_fraction',mode='markers')
dataset.iplot(kind='bar',x='DEATH_EVENT',y='ejection_fraction')
sns.barplot(x='DEATH_EVENT',y='creatinine_phosphokinase',data=dataset)#yes
sns.barplot(x='DEATH_EVENT',y='serum_sodium',data=dataset)#no
sns.violinplot(x='DEATH_EVENT',y='serum_sodium',data=dataset)
sns.barplot(x='DEATH_EVENT',y='anaemia',data=dataset)
dataset.iplot(kind='hist')
dataset_corr = dataset.corr()
fig, ax=plt.subplots(figsize=(15,10))

sns.heatmap(dataset_corr,annot=True)
dataset.columns
X = dataset[['time','ejection_fraction','serum_creatinine']]

y = dataset['DEATH_EVENT']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X_train=ss.fit_transform(X_train)

X_test=ss.transform(X_test)
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(C=0.3,penalty='l1',solver='liblinear')

logistic_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=logistic_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))


from sklearn.metrics import accuracy_score

pre = logistic_model.predict(X_test)
Logistic_acc=accuracy_score(pre,y_test)

print(accuracy_score(pre,y_test))
from sklearn.neighbors import KNeighborsClassifier

score=[]



for i in range(1,10):

    

    

    knn=KNeighborsClassifier(n_neighbors=i,metric='minkowski',p=2)

    knn.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=knn, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre1 = knn.predict(X_test)
knn_acc=accuracy_score(pre1,y_test)

print(accuracy_score(pre1,y_test))
from sklearn.svm import SVC

svm_model=SVC(kernel='rbf',C=0.3,gamma='scale')

svm_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=svm_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre2 = svm_model.predict(X_test)

svm_rbf_acc=accuracy_score(pre2,y_test)

print(accuracy_score(pre2,y_test))
from sklearn.tree import DecisionTreeClassifier

decision_model=DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=3,random_state=30)

decision_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=decision_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre3 = decision_model.predict(X_test)

decision_acc=accuracy_score(pre3,y_test)

print(accuracy_score(pre3,y_test))
from sklearn.ensemble import RandomForestClassifier

randomforest_model=RandomForestClassifier(n_estimators=55,criterion='entropy',random_state=1,max_features=0.5, max_depth=15)

randomforest_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=randomforest_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre4 = randomforest_model.predict(X_test)

random_acc=accuracy_score(pre4,y_test)

print(accuracy_score(pre4,y_test))
from sklearn.svm import SVC

svmlinear_model=SVC(kernel='linear',C=0.1)

svmlinear_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=svmlinear_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre5 = svmlinear_model.predict(X_test)

svm_linear_acc=accuracy_score(pre5,y_test)

print(accuracy_score(pre5,y_test))

from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(X_train,y_train)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=xgb_model, X=X_train ,y=y_train,cv=10)

print("accuracy is {:.2f} %".format(accuracies.mean()*100))

print("std is {:.2f} %".format(accuracies.std()*100))
pre5 = xgb_model.predict(X_test)

xgb_acc=accuracy_score(pre5,y_test)

print(accuracy_score(pre5,y_test))
print("Logistic Accuracy:",Logistic_acc)

print("knn Accuracy:",knn_acc)

print("svm rbf model Accuracy :",svm_rbf_acc)

print("svm linear model Accuracy:",svm_linear_acc)

print("Decision tress Accuracy :",decision_acc)

print("Random_forest _Accuracy:",random_acc)

print("Xgb_boosdt_Accuracy:",xgb_acc)
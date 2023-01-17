import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_table('../input/data-from-uci/processed.cleveland.data',sep=',',names=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target'],na_values='?')
data
df = data.copy()
df
num_cols = ['age','trestbps','chol','thalach','oldpeak']
df['target'].unique()
df['target'] =df['target'].replace(to_replace=[1,2,3,4],value=1)
df.head()
df.isnull().sum()
df['ca'].unique()
df['ca'].value_counts()
df['ca']=df['ca'].fillna(df['ca'].mode()[0])
df['thal'].unique()
df['thal']=df['thal'].fillna(df['thal'].mode()[0])
df.isnull().sum()
df.describe()
df.corr()
plt.pie(x=[(df['target']==0).sum(),(df['target']==1).sum()],labels=[0,1],autopct='%1.2f%%')

print('0 = absence, 1 = presence')
sns.heatmap(df[num_cols].corr(),annot=True)
sns.pairplot(df[num_cols])
sns.barplot(y='age',x='target',data=df)
df[df['target']==0].age.describe()
sns.distplot(df[df['target']==0].age)

plt.title('Age of Patients not having heart disease')
df[df['target']==1].age.describe()
sns.distplot(df[df['target']==1].age)

plt.title('Age of Patients having heart disease')
sns.barplot(y='target',x='sex',data=df)
sns.barplot(x=df['sex'],y=df['age'],hue=df['target'])
sns.countplot(df['cp'])
sns.barplot(x='cp',y='target',data=df)
df['trestbps'].describe()
sns.barplot(x='target',y='trestbps',data=df)
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.distplot(df[df['target']==0].trestbps)

plt.title('Blood Pressure of Patients not having heart disease')

plt.subplot(122)

sns.distplot(df[df['target']==1].trestbps)

plt.title('Blood Pressure of Patients having heart disease')
sns.barplot(x='target',y='chol',data=df)
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.distplot(df[df['target']==0].chol)

plt.title('Cholestrol of Patients not having heart disease')

plt.subplot(122)

sns.distplot(df[df['target']==1].chol)

plt.title('Cholestrol of Patients having heart disease')
sns.countplot(df['fbs'])
sns.barplot(x='target',y='fbs',data=df)
sns.barplot(x='target',y='fbs',data=df)
df['thalach'].describe()
sns.barplot(x='target',y='thalach',data=df)
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.distplot(df[df['target']==0].thalach)

plt.title('Heart Rate of Patients not having heart disease')

plt.subplot(122)

sns.distplot(df[df['target']==1].thalach)

plt.title('Heart Rate of Patients having heart disease')
x = df.drop(['target'],axis=1)
y = df['target']
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler 

sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)

xtest = sc.fit_transform(xtest)
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
dtpred = dt.predict(xtest)
from sklearn.metrics import accuracy_score
dtacc = accuracy_score(ytest,dtpred)*100
print(accuracy_score(ytest,dtpred)*100)
print('Accuracy on Train data {a:.2f}%'.format(a=dt.score(xtrain,ytrain)*100))

print('Accuracy on Test Data {a:.2f}%'.format(a=dt.score(xtest,ytest)*100))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(xtrain,ytrain)
knnpred = knn.predict(xtest)
print(accuracy_score(ytest,knnpred)*100)
knnacc = accuracy_score(ytest,knnpred)*100
print('Accuracy on Train data {a:.2f}%'.format(a=knn.score(xtrain,ytrain)*100))

print('Accuracy on Test Data {a:.2f}%'.format(a=knn.score(xtest,ytest)*100))
score = []



for i in range(1,25):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(xtrain,ytrain)

    knnpred = knn.predict(xtest)

    score.append(accuracy_score(ytest,knnpred))

    
knncc = max(score)*100
knnacc
plt.plot(range(1,25),score)
import xgboost
xgb = xgboost.XGBClassifier()
xgb.fit(xtrain,ytrain)
xgbpred = xgb.predict(xtest)
xgbacc = accuracy_score(ytest,xgbpred)*100
print(accuracy_score(ytest,xgbpred)*100)
print('Accuracy on Train data {a:.2f}%'.format(a=xgb.score(xtrain,ytrain)*100))

print('Accuracy on Test Data {a:.2f}%'.format(a=xgb.score(xtest,ytest)*100))
from sklearn.model_selection import RandomizedSearchCV
params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.4, 0.5 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    

}
rscv = RandomizedSearchCV(xgb,param_distributions=params,n_iter=10,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
rscv.fit(xtrain,ytrain)
rscv.best_estimator_
xgbbp = xgboost.XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.3, gamma=0.1, gpu_id=-1,

              importance_type='gain', interaction_constraints=None,

              learning_rate=0.1, max_delta_step=0, max_depth=5,

              min_child_weight=5, missing=None, monotone_constraints=None,

              n_estimators=100, n_jobs=0, num_parallel_tree=1,

              objective='binary:logistic', random_state=0, reg_alpha=0,

              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,

              validate_parameters=False, verbosity=None)
xgbbp.fit(xtrain,ytrain)
xgbbppred = xgbbp.predict(xtest)
xgbbpacc=accuracy_score(ytest,xgbbppred)*100
print(accuracy_score(ytest,xgbbppred)*100)
print('Accuracy on Train data {a:.2f}%'.format(a=xgbbp.score(xtrain,ytrain)*100))

print('Accuracy on Test Data {a:.2f}%'.format(a=xgbbp.score(xtest,ytest)*100))
dat = { 'Algorithm':['Decision Tree','K Nearest Neighbor','XGBoost','XGB with Optimization'],

        'Accuracy' : [dtacc,knnacc,xgbacc,xgbbpacc]  

       }



pd.DataFrame(dat,columns=['Algorithm','Accuracy'])
models=['Decision Tree','kNN', 'XGBoost', 'XGB with \n Parameter Tuning']

scores=[dtacc,knnacc,xgbacc,xgbbpacc]
sns.barplot(models,scores)
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(10,7))



plt.subplot(221)

plt.title('Decision Tree')

sns.heatmap(confusion_matrix(ytest,dtpred),annot=True)



plt.subplot(222)

plt.title('K Nearest Neighbors')

sns.heatmap(confusion_matrix(ytest,knnpred),annot=True)



plt.subplot(223)

plt.title('XGBoost')

sns.heatmap(confusion_matrix(ytest,xgbpred),annot=True)



plt.subplot(224)

plt.title('XGB with Tuning')

sns.heatmap(confusion_matrix(ytest,xgbbppred),annot=True)
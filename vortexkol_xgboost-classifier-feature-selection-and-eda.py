import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import time
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
col = data.columns       
print(col)
y = data.diagnosis                           
drop_cols = ['Unnamed: 32','id','diagnosis']
x = data.drop(drop_cols,axis = 1 )
x.head()
ax = sns.countplot(y,label="Count")
B, M = y.value_counts()
print('Number of Benign: ',B)
print('Number of Malignant : ',M)
x.describe()
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())              
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
plt.xticks(rotation=45);
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
sns.jointplot(x.loc[:,'concavity_worst'],
              x.loc[:,'concave points_worst'],
              kind="regg",
              color="#ce1414");
#sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())  
data = pd.concat([y,data_n_2.iloc[:,0:10]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,10:20]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=45);
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax);
drop_cols=['perimeter_mean','radius_mean','compactness_mean',
              'concave points_mean','radius_se','perimeter_se',
              'radius_worst','perimeter_worst','compactness_worst',
              'concave points_worst','compactness_se','concave points_se',
              'texture_worst','area_worst']
df=x.drop(drop_cols,axis=1)
df.head()
fig,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=42)

clf_1=xgb.XGBClassifier(random_state=42)
clf_1=clf_1.fit(x_train,y_train)
preds=clf_1.predict(x_test)

print('Accuracy Score :',accuracy_score(preds,y_test))

cm=confusion_matrix(y_test,preds)
sns.heatmap(cm,annot=True,fmt='d')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
select_feature=SelectKBest(chi2,k=10)
select_feature=select_feature.fit(x_train,y_train)
print('Score list:',select_feature.scores_)
print('Features list :',x_train.columns)
x_train_2=select_feature.transform(x_train)
x_test_2=select_feature.transform(x_test)

clf_2=xgb.XGBClassifier()
clf_2=clf_2.fit(x_train_2,y_train)

preds_2=clf_2.predict(x_test_2)

print('Accuracy score =',accuracy_score(preds_2,y_test))

cm_2=confusion_matrix(preds_2,y_test)
sns.heatmap(cm_2,annot=True,fmt='d')

from sklearn.feature_selection import RFECV

clf_3=xgb.XGBClassifier()
rfecv=RFECV(estimator=clf_3,step=1,cv=5,scoring='accuracy',n_jobs=-1).fit(x_train,y_train)

print('Optimal features =',rfecv.n_features_)
print(' Best features =',x_train.columns[rfecv.support_])
accuracy_score(y_test,rfecv.predict(x_test))
num_features=[i for i in range(1,len(rfecv.grid_scores_)+1)]
cv_scores=rfecv.grid_scores_
ax=sns.lineplot(x=num_features,y=cv_scores)
ax.set(xlabel='No.of selected features',ylabel='CV_Scores')
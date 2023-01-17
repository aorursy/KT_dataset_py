import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

color=sns.color_palette()
data=pd.read_csv('../input/heart.csv')

data.head()
fig,ax=plt.subplots(1, 2, figsize = (14,5))

sns.countplot(data=data, x='target', ax=ax[0],palette='Set2')

ax[0].set_xlabel("Підрахунок захворювань \n [0]->Ні [1]->Так")

ax[0].set_ylabel("Кількість")

ax[0].set_title("Кількість серцевих захворювань")

data['target'].value_counts().plot.pie(explode=[0.1,0.0],autopct='%1.1f%%',ax=ax[1],shadow=True, cmap='Greens')

plt.title("Захворювання серця")
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='sex',data=data,hue='target',palette='Set1',ax=ax[0])

ax[0].set_xlabel("0 ->Female , 1 ->Male")

data.sex.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0], cmap='Reds')

ax[1].set_title("0 ->Female , 1 -> Male")
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='fbs',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("0-> fps <120 , 1-> fps>120",size=12)

data.fbs.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True, explode=[0.1,0],cmap='Oranges')

ax[1].set_title("0 -> fps <120 , 1 -> fps>120",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='restecg',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("resting electrocardiographic",size=12)

data.restecg.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,

                                     explode=[0.005,0.05,0.05],cmap='Blues')

ax[1].set_title("resting electrocardiographic",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='slope',data=data,hue='target',palette='Set1',ax=ax[0])

ax[0].set_xlabel("peak exercise ST segment",size=12)

data.slope.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,explode=[0.005,0.05,0.05],cmap='Blues')



ax[1].set_title("peak exercise ST segment ",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='ca',data=data,hue='target',palette='Set2',ax=ax[0])

ax[0].set_xlabel("number of major vessels colored by flourosopy",size=12)

data.ca.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',shadow=True,cmap='Oranges')

ax[1].set_title("number of major vessels colored by flourosopy",size=12)
fig,ax=plt.subplots(1,2,figsize=(14,5))

sns.countplot(x='cp',data=data,hue='target',palette='Set3',ax=ax[0])

ax[0].set_xlabel("Chest Pain")

data.cp.value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0.01,0.01,0.01,0.01],shadow=True, cmap='Blues')

ax[1].set_title("Chest pain")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='trestbps',data=data,x='sex',hue='target',palette='Set2',ax=ax[0,0])

ax[0,0].set_title("Trestbps V/S Sex")

sns.factorplot(y='trestbps',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set2')

ax[0,1].set_title("Trestbps V/S Chest Pain")

sns.violinplot(y='trestbps',data=data,x='exang',hue='target',ax=ax[1,0],palette='Set2')

ax[1,0].set_title("Trestbps V/S Exang")

sns.swarmplot(y='trestbps',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set2')

ax[1,1].set_title("Trestbps V/S CA (Major Vessel Coloured)")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='chol',data=data,x='sex',hue='target',palette='Set3',ax=ax[0,0])

ax[0,0].set_title("Cholestrol V/S Sex")

sns.boxplot(y='chol',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set3')

ax[0,1].set_title("Cholestrol V/S Chest Pain")

sns.swarmplot(y='chol',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set3')

ax[1,0].set_title("Cholestrol V/S Thal")
fig,ax=plt.subplots(2,2,figsize=(14,10))

sns.boxplot(y='oldpeak',data=data,x='sex',hue='target',palette='Set1',ax=ax[0,0])

ax[0,0].set_title("oldpeak V/S Sex")

sns.boxplot(y='oldpeak',data=data,x='cp',hue='target',ax=ax[0,1],palette='Set1')

ax[0,1].set_title("oldpeak V/S Chest Pain")

sns.swarmplot(y='oldpeak',data=data,x='thal',hue='target',ax=ax[1,0],palette='Set1')

ax[1,0].set_title("oldpeak V/S Thal")

sns.factorplot(y='oldpeak',data=data,x='ca',hue='target',ax=ax[1,1],palette='Set1')

ax[1,1].set_title("oldpeak V/S CA")
fig,ax=plt.subplots(4,3,figsize=(15,15))

for i in range(12):

    plt.subplot(4,3,i+1)

    sns.distplot(data.iloc[:,i],kde=True, color='green')
fig,ax=plt.subplots(figsize=(15,5))

sns.heatmap(data.isnull(), annot=True)
fig=plt.figure(figsize=(18,18))

sns.heatmap(data.corr(), annot= True, cmap='Blues')
data.sex=data.sex.astype('category')

data.cp=data.cp.astype('category')

data.fbs=data.fbs.astype('category')

data.restecg=data.restecg.astype('category')

data.exang=data.exang.astype('category')

data.ca=data.ca.astype('category')

data.slope=data.slope.astype('category')

data.thal=data.thal.astype('category')
data_label=data['target']

del data['target']

data_label=pd.DataFrame(data_label)
data=pd.get_dummies(data,drop_first=True)

data.head(),data_label.head()
from sklearn.preprocessing import StandardScaler

data_scaled=StandardScaler().fit_transform(data)

data_scaled=pd.DataFrame(data=data_scaled, columns=data.columns)
data_scaled.head()
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(data_scaled, data_label, test_size=0.20,

                                             stratify=data_label,random_state=975456)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.model_selection import cross_val_score
def CrossVal(dataX,dataY,mode,cv=10):

    score=cross_val_score(mode,dataX , dataY, cv=cv, scoring='accuracy')

    return(np.mean(score))
def plotting(true,pred):

    fig,ax=plt.subplots(1,2,figsize=(10,5))

    precision,recall,threshold = precision_recall_curve(true,pred[:,1])

    ax[0].plot(recall,precision,'g--')

    ax[0].set_xlabel('Recall')

    ax[0].set_ylabel('Precision')

    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))

    fpr,tpr,threshold = roc_curve(true,pred[:,1])

    ax[1].plot(fpr,tpr)

    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))

    ax[1].plot([0,1],[0,1],'k--')

    ax[1].set_xlabel('False Positive Rate')

    ax[1].set_ylabel('True Positive Rate')
dtc=DecisionTreeClassifier(max_depth=7)

score_dtc=CrossVal(Xtrain,Ytrain,dtc)

print("Accuracy is : ",score_dtc)

dtc.fit(Xtrain,Ytrain)

plotting(Ytest,dtc.predict_proba(Xtest))



rf=RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=253)

score_rf= CrossVal(Xtrain,Ytrain,rf)

print('Accuracy is:',score_rf)

rf.fit(Xtrain,Ytrain)

plotting(Ytest,rf.predict_proba(Xtest))



etc=ExtraTreesClassifier(n_estimators=10, n_jobs=-1, random_state=800)

score_etc= CrossVal(Xtrain,Ytrain,etc)

print('Accuracy is:',score_etc)

etc.fit(Xtrain,Ytrain)

plotting(Ytest,etc.predict_proba(Xtest))



fig= plt.figure(figsize=(10,10))

important=pd.Series(rf.feature_importances_, index=Xtrain.columns)

sns.set_style('whitegrid')

important.sort_values().plot.barh()

plt.title('Feature Importance')
model_accuracy = pd.Series(data=[score_dtc, score_rf, score_etc, ], 

                           index=['decision tree','Random Forest', 'Extra Tree'])

fig= plt.figure(figsize=(8,8))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')
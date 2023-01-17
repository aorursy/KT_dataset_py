# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
from matplotlib import pyplot as plt

import seaborn as sn

from dateutil import parser

import time
dataset=pd.read_csv('../input/financeapp/p39-cs3-data/appdata10.csv')

dataset.head()
dataset.describe()
dataset['hour']=dataset.hour.str.slice(1,3).astype(int)
dataset.head()
dataset2=dataset.copy().drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])
dataset2.head()
plt.suptitle('Histogram of Numerical Column', fontsize=20)

for i in range(1,dataset2.shape[1]+1):

    plt.figure(num=7,figsize=(20,30))

    plt.subplot(7,2,i)

    f=plt.gca()

    f.set_title(dataset2.columns.values[i-1])

    vals=np.size(dataset2.iloc[:,i-1].unique())

    plt.hist(dataset2.iloc[:,i-1],bins=vals,color='#3f5d7d')
dataset2.corrwith(dataset.enrolled).plot.bar(figsize=(20,10),

                                                title='Correlation with Response Variable',

                                                fontsize=15,rot=45,grid=True

                                               )
sn.set(style="white",font_scale=2)

corr=dataset2.corr()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

f,ax=plt.subplots(figsize=(18,15))

f.suptitle("Correlation Matrix",fontsize=40)

cmap=sn.diverging_palette(220,10,as_cmap=True)

sn.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=5,cbar_kws={"shrink":.5})

from dateutil import parser

dataset.dtypes


dataset["first_open"] = [parser.parse(str(row_date)) for row_date in dataset["first_open"]]

dataset["enrolled_date"] = [parser.parse(str(row_date)) if isinstance(row_date, str) else row_date for row_date in dataset["enrolled_date"]]

dataset.dtypes
dataset["diffrences"]=(dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
plt.figure(num=1,figsize=(20,10))

plt.hist(dataset.diffrences.dropna(),color="#3f5d7d")

plt.title("Distribution of Time-Since-Enrolled")

plt.show()
plt.figure(figsize=(20,10))

plt.hist(dataset.diffrences.dropna(),color="#3f5d7d",range=[0,100])

plt.title("Distribution of Time-Since-Enrolled")

plt.show()
dataset.loc[dataset.diffrences>48 ,"enrolled"]=0

dataset=dataset.drop(columns=["diffrences","enrolled_date","first_open"])
top_screen=pd.read_csv("../input/financeapp/p39-cs3-data/top_screens.csv").top_screens.values

dataset["screen_list"]=dataset.screen_list.astype(str) +','
for sc in top_screen:

    dataset[sc]=dataset.screen_list.str.contains(sc).astype(int)

    dataset["screen_list"]=dataset.screen_list.str.replace(sc+",","")
dataset.head()
dataset["Other"]=dataset.screen_list.str.count(",")

dataset=dataset.drop(columns=["screen_list"])
dataset.head()
savings_screens = ["Saving1",

                    "Saving2",

                    "Saving2Amount",

                    "Saving4",

                    "Saving5",

                    "Saving6",

                    "Saving7",

                    "Saving8",

                    "Saving9",

                    "Saving10"]

dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)

dataset = dataset.drop(columns=savings_screens)
cm_screens = ["Credit1",

               "Credit2",

               "Credit3",

               "Credit3Container",

               "Credit3Dashboard"]

dataset["CMCount"] = dataset[cm_screens].sum(axis=1)

dataset = dataset.drop(columns=cm_screens)
cc_screens = ["CC1",

                "CC1Category",

                "CC3"]

dataset["CCCount"] = dataset[cc_screens].sum(axis=1)

dataset = dataset.drop(columns=cc_screens)
loan_screens = ["Loan",

               "Loan2",

               "Loan3",

               "Loan4"]

dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)

dataset = dataset.drop(columns=loan_screens)
dataset.head()
response=dataset.enrolled

dataset=dataset.drop(columns=["enrolled"])

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(dataset,response,test_size=0.05,random_state=0)
train_identifier=X_train["user"]

X_train=X_train.drop(columns="user")

test_identifier=X_test["user"]

X_test=X_test.drop(columns="user")

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train2=pd.DataFrame(sc.fit_transform(np.float64(X_train)))

X_test2=pd.DataFrame(sc.transform(np.float64(X_test)))

X_train2.columns=X_train.columns.values

X_test2.columns=X_test.columns.values

X_train2.index=X_train.index.values

X_test2.index=X_test2.index.values

X_train=X_train2

X_test=X_test2

X_train.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,classification_report

from xgboost import XGBClassifier
model = [GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(n_estimators=5, random_state=0,), LogisticRegression(random_state=0,penalty='l1',solver='saga'),XGBClassifier(),SVC(gamma='auto')]

model_names = ["Gaussian Naive bayes", "K-nearest neighbors", "Decision tree classifier", "Random Forest", "Logistic Regression"," XGBoost","Support vector classifier"]

for i in range(0, 7):

    y_pred = model[i].fit(X_train, y_train).predict(X_test)

    accuracy = accuracy_score(y_pred, y_test)*100

    print(model_names[i], ":", accuracy, "%")

    print(classification_report(y_test,y_pred))

    print('-'*70)

    
classifier=SVC(gamma='auto')

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,classification_report
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
df_cm=pd.DataFrame(cm,index=(0,1),columns=(0,1))

plt.figure(figsize=(10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm,annot=True,fmt='g')
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean(),accuracies.std()
#Formatting the final result for ML

final_result=pd.concat([y_test,test_identifier],axis=1)

final_result["predicted_result"]=y_pred

final_result.head(10)
final_result[final_result.enrolled!=final_result.predicted_result].head(10)
"""feature_imp=pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),

           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])

           ],axis = 1)"""
#feature_imp
from keras.models import Sequential

from keras.layers import Dense,Dropout,BatchNormalization

from keras.callbacks import  ModelCheckpoint
model=Sequential()

model.add(Dense(32,activation='relu',input_dim=48,kernel_initializer='uniform'))

model.add(BatchNormalization())

model.add(Dense(16,activation='relu',kernel_initializer='uniform'))

model.add(BatchNormalization())

model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#checkpoint = ModelCheckpoint('{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#model.fit(X_train,y_train,batch_size=32,epochs=50,shuffle=True,verbose=1, callbacks=[checkpoint],validation_data=(X_test,y_test))
model.load_weights('../input/customersuscription/0.7916.hdf5')
y_pred_dl=model.predict(X_test)

y_pred_dl_prob=model.predict_proba(X_test)
print(classification_report(y_test,np.round(y_pred_dl)))
accuracy_score(y_test,np.round(y_pred_dl))
cm_dl=confusion_matrix(y_test,np.round(y_pred_dl))
df_cm_dl=pd.DataFrame(cm_dl,index=(0,1),columns=(0,1))

plt.figure(figsize=(10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm_dl,annot=True,fmt='g')
#Formatting the final result for DL

final_result_dl=pd.concat([y_test,test_identifier],axis=1)

final_result_dl["predicted_result"]=y_pred_dl

final_result_dl["Probabilty"]=[max(result) for result in y_pred_dl_prob]

final_result_dl.head(10)
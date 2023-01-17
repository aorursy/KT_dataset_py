import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings('ignore')



from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score



from sklearn.model_selection import train_test_split



df = pd.read_csv("../input/krediVeriseti.csv", sep = ";")

df.head()
df.evDurumu[df.evDurumu == 'evsahibi'] = 1

df.evDurumu[df.evDurumu == 'kiraci'] = 0

#df['evDurumu'].value_counts()

df.telefonDurumu[df.telefonDurumu == 'var'] = 1

df.telefonDurumu[df.telefonDurumu == 'yok'] = 0

#df['telefonDurumu'].value_counts()

df.KrediDurumu[df.KrediDurumu == 'krediver'] = True

df.KrediDurumu[df.KrediDurumu == 'verme'] = False

#df['KrediDurumu'].value_counts()
df.tail()
df = df.astype(float)
df.describe()
df.info()
df.head(5)
df.isnull().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
X=df.drop('KrediDurumu',axis=1)

Y=df['KrediDurumu']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
X
Y
print ("X_train: ", len(x_train))

print("X_test: ", len(x_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

log=LogisticRegression(C=0.0007)

log.fit(x_train,y_train)

print('Logistic Regression Train score:',log.score(x_train,y_train))

print('Logistic Regression Test score:',log.score(x_test,y_test))

log_y_pred = log.predict(x_test)
log_score = f1_score(y_test,log_y_pred)

print('Logistic Regression Score(f1): ', f1_score(y_test,log_y_pred))

print('Logistic Regression Score(r2): ', r2_score(y_test,log_y_pred))

print('Logistic Regression Score(acc): ', accuracy_score(y_test,log_y_pred))

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)

print('Decision Tree Train score:',dt.score(x_train,y_train))

print('Decision Tree Test score:',dt.score(x_test,y_test))

dt_y_pred = dt.predict(x_test)
dt_score = f1_score(y_test,dt_y_pred)

print('Decision Tree Score(f1): ', f1_score(y_test,dt_y_pred))

print('Decision Tree Score(r2): ', r2_score(y_test,dt_y_pred))

print('Decision Tree Score(acc): ', accuracy_score(y_test,dt_y_pred))
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

rf=RandomForestClassifier()

rf.fit(x_train,y_train)

print('Random Forest Train score:',rf.score(x_train,y_train))

print('Random Forest Test score:',rf.score(x_test,y_test))

rf_y_pred = rf.predict(x_test)
rf_score = f1_score(y_test,rf_y_pred)

print('Random Forest Score(f1): ', f1_score(y_test,rf_y_pred))

print('Random Forest Score(r2): ', r2_score(y_test,rf_y_pred))

print('Random Forest Score(acc): ', accuracy_score(y_test,rf_y_pred))
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

print('Naive Bayes Train score:',nb.score(x_train,y_train))

print('Naive Bayes Test score:',nb.score(x_test,y_test))

nb_y_pred = nb.predict(x_test)
nb_score = f1_score(y_test,nb_y_pred)

print('Naive Bayes Score(f1): ', f1_score(y_test,nb_y_pred))

print('Naive Bayes Score(r2): ', r2_score(y_test,nb_y_pred))

print('Naive Bayes Score(acc): ', accuracy_score(y_test,nb_y_pred))
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)

knn.fit(x_train,y_train)

print('KNNTrain score:',knn.score(x_train,y_train))

print('KNN Test score:',knn.score(x_test,y_test))

knn_y_pred = knn.predict(x_test)

knn_score = f1_score(y_test,knn_y_pred)

print('KNN Score(f1): ', f1_score(y_test,knn_y_pred))

print('KNN Score(r2): ', r2_score(y_test,knn_y_pred))

print('KNN Score(acc): ', accuracy_score(y_test,knn_y_pred))
PredictionTable = pd.DataFrame({'Gerçek':y_test.ravel(),

                                'LR Tahmin':log_y_pred.ravel(),

                               'DT Tahmin':dt_y_pred.ravel(),

                               'RF Tahmin':rf_y_pred.ravel(),

                               'NB Tahmin':nb_y_pred.ravel(),

                               'KNN Tahmin':knn_y_pred.ravel()})

PredictionTable.sample(10)
print('LR:',log_score)

print('DT:',dt_score)

print('RF:',rf_score)

print('NB:',nb_score)

print('KNN:',knn_score)



test = log.predict([[5951,22,1,1,0]])

print(test)



if(round(test[0])>=1):

    print("Kredi verilebilir")

else:

    print("Kredi verilemez")  
import pickle

pickle.dump(log, open('myModel.pkl','wb'))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df=pd.DataFrame(pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv'))

df.info()
#Remove na

df=df.drop(labels='company',axis='columns')

df=df.dropna(axis=0)

df.info()
#Replace Months

df['arrival_date_month']=df['arrival_date_month'].map({

    'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,'August':8, 'September':9, 'October':10, 'November':11, 'December':12

})

df['arrival_date_month'].unique()
#Replace Other Object Dtypes

cat_col=df.columns[df.dtypes=='O']

cat_col
from sklearn import preprocessing



OE=preprocessing.OrdinalEncoder()

df[cat_col]=OE.fit_transform(df[cat_col])
df.info()
#Map Correlations

import matplotlib.pyplot as plt



dfc=df.corr()

fig,ax=plt.subplots(figsize=(10,10))

im=ax.imshow(dfc,vmax=1,vmin=-1)

plt.colorbar(im,shrink=0.8)

ax.set_xticks(np.arange(len(dfc.columns)))

ax.set_xticklabels(dfc.columns,rotation=90,size='x-large')

ax.set_yticks(np.arange(len(dfc.index)))

ax.set_yticklabels(dfc.index,size='x-large')

im.set(cmap='PiYG')
#Predict potential cancellation

from sklearn import model_selection

from sklearn import linear_model

from sklearn import metrics



x_train,x_test,y_train,y_test=model_selection.train_test_split(df.loc[:,df.columns!='is_canceled'],df['is_canceled'],test_size=0.2,random_state=0)

LogRe=linear_model.LogisticRegression()

LogRe.fit(x_train,y_train)

y_predict=LogRe.predict(x_test)

print('F1 Score: ',metrics.f1_score(y_test,y_predict))

print('R2 Score: ',metrics.r2_score(y_test,y_predict))
from sklearn import ensemble



RFR=ensemble.RandomForestClassifier()

RFR.fit(x_train,y_train)

y_predict=RFR.predict(x_test)

print('F1 Score: ',metrics.f1_score(y_test,y_predict))

print('R2 Score: ',metrics.r2_score(y_test,y_predict))
from sklearn import svm



LinSVC=svm.LinearSVC()

LinSVC.fit(x_train,y_train)

y_predict=LinSVC.predict(x_test)

print('F1 Score: ',metrics.f1_score(y_test,y_predict))

print('R2 Score: ',metrics.r2_score(y_test,y_predict))
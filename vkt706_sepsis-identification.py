import matplotlib.pyplot as plt

from os import listdir

import seaborn as sns

import time

from pylab import rcParams

from os import listdir

import csv

from xgboost import XGBClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import accuracy_score



a_traincsv = listdir('/kaggle/input/training_setA/training')

b_traincsv = listdir('/kaggle/input/training_setB/training_setB')
with open('train_patient.csv', 'w') as csvoutput:

  writer = csv.writer(csvoutput, lineterminator='\n')



  for ind, csv_name in enumerate(a_traincsv):

    with open('/kaggle/input/training_setA/training/'+ csv_name,'r') as csvinput:

      reader = csv.reader(csvinput, delimiter='|')

      all = []

      if ind ==0 :

        row = next(reader)

        row.append('Patient_id')

        row.append('time')

        all.append(row)

      else:

        row = next(reader)



      for i,row in enumerate(reader):

        row.append(ind)

        row.append(i)

        all.append(row)

      writer.writerows(all)

    

  num = ind 

  for inde, csv_name in enumerate(b_traincsv):

    if inde < 10000:

      num = num+1

      with open('/kaggle/input/training_setB/training_setB/'+ csv_name,'r') as csvinput:

        reader = csv.reader(csvinput, delimiter='|')

        all = []

        row = next(reader)

        for i,row in enumerate(reader):

          row.append(num)

          row.append(i)

          all.append(row)

        

        writer.writerows(all)

        



with open('val_patient.csv', 'w') as csvoutput:

  writer = csv.writer(csvoutput, lineterminator='\n')



  for inde, csv_name in enumerate(b_traincsv):

    if inde >=10000 and inde <15000:

      with open('/kaggle/input/training_setB/training_setB/'+ csv_name,'r') as csvinput:

        reader = csv.reader(csvinput, delimiter='|')

        all = []

        if inde ==10000 :

          row = next(reader)

          row.append('Patient_id')

          row.append('time')

          all.append(row)

        else:

          row = next(reader)

        for i,row in enumerate(reader):

          row.append(inde)

          row.append(i)

          all.append(row)

        writer.writerows(all)

        



with open('test_patient.csv', 'w') as csvoutput:

  writer = csv.writer(csvoutput, lineterminator='\n')



  for inde, csv_name in enumerate(b_traincsv):

    if inde >=15000:

      with open('/kaggle/input/training_setB/training_setB/'+ csv_name,'r') as csvinput:

        reader = csv.reader(csvinput, delimiter='|')

        all = []

        if inde ==15000 :

          row = next(reader)

          row.append('Patient_id')

          row.append('time')

          all.append(row)

        else:

          row = next(reader)

        for i,row in enumerate(reader):

          row.append(inde)

          row.append(i)

          all.append(row)



        writer.writerows(all)
train = pd.read_csv('train_patient.csv')

train.head()
train_copy = train.groupby('Patient_id').mean().fillna(train.mean())

train_copy['SepsisLabel'][train_copy['SepsisLabel']!=0] = 1

x = train_copy.drop(['SepsisLabel'],axis = 1)

y = train_copy['SepsisLabel']
test = pd.read_csv('test_patient.csv')

test_copy = test.fillna(train.mean())

x_test = test_copy.drop(['SepsisLabel','Patient_id'],axis = 1)

y_test = test_copy['SepsisLabel']
val = pd.read_csv('val_patient.csv')

val = val.fillna(train.mean())

x_val = val.drop(['SepsisLabel','Patient_id'],axis = 1)

y_val = val['SepsisLabel']
y.value_counts()
xgb = XGBClassifier()

xgb.fit(x,y)



y_pred = xgb.predict(x)

print("Train Accuracy: ",accuracy_score(y_pred, y))

y_pred = xgb.predict(x_test)

print("Test Accuracy: ",accuracy_score(y_pred, y_test))

y_pred = xgb.predict(x_val)

print("Validation Accuracy: ",accuracy_score(y_pred, y_val))
from sklearn import metrics 

y_pred = xgb.predict(x_test)

print("Test Accuracy: ",accuracy_score(y_pred, y_test))

cm=metrics.confusion_matrix(y_test, y_pred)

print(cm)

import seaborn as sn

sn.heatmap(cm, annot=True)
corrmat = train.corr()['SepsisLabel']

  

#f, ax = plt.subplots(figsize =(9, 8)) 

corrmat#.style.background_gradient(cmap='coolwarm')
pd.concat([train[train['SepsisLabel'] == 0][:int(1172238*0.02)],train[train['SepsisLabel'] == 1]],axis = 0).corr()['SepsisLabel']
train[train['Patient_id'] == 8]
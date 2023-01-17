!pip install tensorly
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import csv

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import tensorflow as tf        

import tensorly       

from tensorly.decomposition import non_negative_parafac

from tensorly.metrics import regression

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
userdata = pd.read_csv('../input/credit-card-approval-prediction/application_record.csv')

recorddata = pd.read_csv('../input/credit-card-approval-prediction/credit_record.csv')

print(recorddata)
userdata['STATUS']= recorddata['STATUS']

userdata = userdata.loc[0:5000,:] 

print(userdata.shape)

row= userdata.shape[0]

print(row)

a = [1,2,3]



#userdata['creditType'] = [ "Bad" if (x == '1' or x == '2' or x == '3' or x == '4' or x == '5') else "Good" for x in userdata.STATUS]

userdata['Status'] = [ 1 if (x == '0') else 1 for x in userdata.STATUS]



for i,j in userdata.iterrows():

    if j.STATUS == '1':

        j.Status = 1 

    elif j.STATUS == '2':

        j.Status = 2 

    elif j.STATUS == '3':

        j.Status = 3 

    elif j.STATUS == '4':

        j.Status = 4 

    elif j.STATUS == '5':

        j.Status = 5 

        

#randomly placing numbers in status

for i in range(0,100):

    x = random.randint(1,5)

    rowno = random.randint(0,5000)

    userdata.iloc[rowno,19] = x



headers=list(userdata.columns)

print(headers)

a=list(userdata.loc[0])

print(a)

userdata.to_csv('csv_to_submit.csv', index = False)

df=userdata[['AMT_INCOME_TOTAL','FLAG_OWN_CAR','DAYS_EMPLOYED','CNT_FAM_MEMBERS']].copy()







df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED'].astype('category').cat.codes

df['FLAG_OWN_CAR']=df['FLAG_OWN_CAR'].astype('category').cat.codes

#df['FLAG_OWN_REALTY']=df['FLAG_OWN_REALTY'].astype('category').cat.codes

#df['NAME_EDUCATION_TYPE']=df['NAME_EDUCATION_TYPE'].astype('category').cat.codes

#df['NAME_FAMILY_STATUS']=df['NAME_FAMILY_STATUS'].astype('category').cat.codes

#df['NAME_HOUSING_TYPE']=df['NAME_HOUSING_TYPE'].astype('category').cat.codes

#df['STATUS']=df['STATUS'].astype('category').cat.codes





df.corr(method ='kendall') 
from sklearn.decomposition import PCA 

pca = PCA(n_components=1)

principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(principalComponents, columns = ['Economic_Indicator'])

principalDf['ID']=userdata['ID']

principalDf['Status']= userdata['Status']

principalDf['OCCUPATION_TYPE']= userdata['OCCUPATION_TYPE']

print(principalDf['Economic_Indicator'])
l = list(principalDf['OCCUPATION_TYPE'])

m = list(principalDf['ID'])

n = list(principalDf['Economic_Indicator'])





Occupation = dict([(y,x+1) for x,y in enumerate(set(l))])

userlist = dict([(y,x+1) for x,y in enumerate(set(m))])

eco_in = dict([(int(y),x+1) for x,y in enumerate(set(n))])



all_values = eco_in.values()

max_value = max(all_values) + 1



t = tf.ones([20,max_value,5002])

tensor = tensorly.tensor(t) 

print(eco_in)



print(tensor[0,0,0])







indexlist = []

valuelist = []



for i,j in principalDf.iterrows():

    indexlist.append((Occupation[j.OCCUPATION_TYPE],eco_in[int(j.Economic_Indicator)], userlist[j.ID]))

    valuelist.append(float(j.Status))

    

print(indexlist)

print(valuelist)



indices = tf.constant(indexlist)

updates = tf.constant(valuelist)

print(tf.tensor_scatter_nd_update(tensor, indices, updates))

print(type(tensor))

R = 3

g = tf.keras.utils.normalize(tensor, axis=2, order=2)



print(type(g))

print(g.shape)

tensor2 = tensorly.tensor(g)

print(tensor2.shape)

factors = non_negative_parafac(tensor2,rank = 3, verbose = 2)

print(factors)
print(len(factors))
print(factors[0].shape)

print(factors[1][0].shape)

print(factors[1][1].shape)

print(factors[0])

print(factors[1][0])

print(factors[1][1])
regenerated_tensor = tensorly.kruskal_to_tensor(factors)

print(regenerated_tensor.shape)
#Error measurement

print("Root Mean Square error = "+str(tensorly.metrics.regression.RMSE(tensor, regenerated_tensor, axis=None)))

print("Mean squared error = " + str(tf.keras.losses.MSE(tensor,regenerated_tensor)))

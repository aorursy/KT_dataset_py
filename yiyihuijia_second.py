import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data=pd.read_csv("../input/train.csv",usecols=['Producto_ID','Demanda_uni_equil'])

train_data['log_Dem']=np.log(np.array(train_data['Demanda_uni_equil'].tolist())+1)

#print(train_data)
mean_data=train_data.groupby(train_data['Producto_ID']).mean()

print(mean_data)
test_data=pd.read_csv("../input/test.csv",usecols=['id','Producto_ID'])

target=np.zeros(test_data.shape[0])

log_target=np.zeros(test_data.shape[0])

for pid in mean_data.index:

    target[test_data[test_data['Producto_ID']==pid]['id'].values]=mean_data.ix[pid]['Demanda_uni_equil']

    log_target[test_data[test_data['Producto_ID']==pid]['id'].values]=mean_data.ix[pid]['log_Dem']

#print (log_target)



test_data['Demanda_uni_equil']=np.exp(log_target)-1

print(test_data)

test_data.to_csv('result_groupmean_log.csv',index=False,columns=['id','Demanda_uni_equil'])
test_data[test_data['Producto_ID']==41]['id']
#mean_data.index

#mean_data.ix[41]

test_data.shape
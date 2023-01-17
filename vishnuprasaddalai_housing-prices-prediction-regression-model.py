# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import  numpy as np
test_set = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
dataset = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
tset = []
x11 = dataset.iloc[:,:-1].values
x22 = test_set.iloc[:,:].values
tset.append(x11)
tset.append(x22)
tset = np.concatenate(tset)
print(tset)
#information
print('INFORMATION')
print('dataset length: ',len(tset[0]))
print('Dataset Column Names and Type: ')
for i in range(0,len(tset[0])):
    print(dataset.columns[i], ' ', type(dataset.values[0][i]))
check_set = []
check_save = []
check_no = []
for i in range(0,len(tset[0])-1):
    for j in range(0,len(tset[0])):
        check_set.append(tset[j][i])
    if len(list(set(check_set)))<50:
        print('Categorical: ',i, '| No. Categories: ',len(list(set(check_set))),'| Categories: ',list(set(check_set)))
        check_no.append(i)
        check_save.append(list(set(check_set)))
    
    check_set.clear()
y = dataset.iloc[:,-1].values
#replace nan by mean for numerical values
from sklearn.impute import SimpleImputer
si1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
check_nan = []
for i in range(len(tset[0])-1):
    if i not in check_no and type(tset[0][i]):
        print(i, ' Done, Type: ',type(tset[0][i]))
        tset[:,i:i+1] = si1.fit_transform(tset[:,i:i+1])
print('Training set: ',tset[len(dataset.Id)-1,0:80])
print('Test Set: ',tset[len(dataset.Id)-1,0:80])
#Fill NAN values by most frequent for strings

si = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
for i in range(len(tset[0])-1):
    if i in check_no and type(tset[0][i]) == str:
        
        tset[:,i:i+1] = si.fit_transform(tset[:,i:i+1])
            
#Encoding categorical values using One Hot Encoder
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
lc = LabelEncoder()
xx = []
for i in range(0,len(check_no)):

    tset[:,check_no[i]] = lc.fit_transform(tset[:,check_no[i]].astype(str))
    
    
    
    
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(tset)

xx1 = np.array(onehot_encoded)
xtrainf = xx1[:len(dataset.Id),:]
xtestf = xx1[len(dataset.Id):,:]
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 500, random_state = 0)
reg.fit(xtrainf,y)
ypred_f = reg.predict(xtestf)
id_test = []
for i in range(0,len(test_set.Id)):
    id_test.append(i+1461)

#predicted prices
df = pd.DataFrame({'Id': id_test, 'SalePrice': ypred_f })
print(df)
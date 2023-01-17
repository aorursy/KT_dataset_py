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
import numpy as np
import pandas as pd
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from heapq import nsmallest
import itertools
import random
from random import randrange
import time
data = pd.read_csv('../input/3-montth/3 montth.csv')
data.info()
data.head()
X = data.drop('102',axis=1)
Y = data['102']
Y
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# fit model to training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted')*100.0)
data2=pd.DataFrame(columns=['y_test','y_pred'])
data2['y_test']=y_test
data2['y_pred']=y_pred
data2
result_unique = pd.DataFrame()

print(X_train.head())
print(data['52'])
column_values = data[["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                      "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
                      "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", 
                      "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", 
                      "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", 
                      "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", 
                      "96", "97", "98", "99", "100","101", "102"]].values.ravel()
unique_values =  pd.unique(column_values)
UNfb=pd.DataFrame(unique_values)
UNfb
result_unique['u_val']=unique_values
from heapq import nsmallest,nlargest
result = pd.DataFrame()
result['predicted'] = y_pred
result['Y_test'] = y_test.tolist()
result[result['predicted']==result['Y_test']]
encoded_list = []
encoded_list.append(nlargest(30, result_unique['u_val'], key=lambda x: abs(x-result['predicted'].iloc[0])))
print(result['predicted'].iloc[0])
print(encoded_list)
flag_counter=0
encoded_list = []
for i in range(len(result)):
    encoded_list.append(nsmallest(150, result_unique['u_val'], key=lambda x: abs(x-result['predicted'].iloc[i])))
    #print(result['predicted'].iloc[i])
    #print(nsmallest(5, result['Y_test'], key=lambda x: abs(x-result['predicted'].iloc[i])))
print(len(encoded_list))
counter=0
#print (result['predicted'].iloc[counter])
label_list = []
p_list=[]


for i in encoded_list:
    flag=0
    p_val=result['Y_test'].iloc[counter]
    #p_list.append(i)
    list_item=[]
    for j in range(len(i)):
        if p_val==i[j]:
            list_item.extend((i[j-4],i[j-3],i[j-2],i[j-1],i[j]))
            #list_item.sort()   
            p_list.append(list_item)  
            flag=1
    if flag==1:
        #print("True")
        label_list.append("True")
    else:
        #print("False")
        f_val=5
        #list_item.extend((result['Y_test'].iloc[a],result['Y_test'].iloc[b],result['Y_test'].iloc[c],result['Y_test'].iloc[d],result['Y_test'].iloc[counter]))
        list_item.extend((i[j-4],i[j-3],i[j-2],i[j-1],i[j]))
        #list_item.sort()   
        p_list.append(list_item)
        label_list.append("False")
    counter=counter+1
count=0
for k in label_list:
    if k=="True":
        count=count+1
print("Counter:", count)
print(len(y_pred))
print(len(p_list))
print(len(label_list))
result["P_Values"]=p_list
result["T_F"]=label_list
print(result)
print(result[(result['T_F'] == 'True')])
print("Pattern Recoginition:",len(result[result['T_F']=='True'])/len(result)*100)
# f = plt.figure(figsize=(19, 15))
# plt.matshow(data.corr(), fignum=f.number)
# plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
# plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
import pandas as pd
import numpy as np

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r' & 'BrBG' are other good diverging colormaps
data = pd.read_csv('../input/3-monttth/3 monttth.csv') 
X1 = data.drop('wwww',axis=1)
data.info()
Y1 = data['wwww']
print(Y1)
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2, random_state=seed)
print(len(y_test))
# fit model to training data
model = XGBClassifier(num_class = len(y_train.unique()),objective='multi:softmax',)
eval_set_val = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_train, y_train)], verbose=True)
#m2=model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=[(X_test, y_test)], verbose=True)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
results = model.evals_result()
print(results)
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)
from matplotlib import pyplot

# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
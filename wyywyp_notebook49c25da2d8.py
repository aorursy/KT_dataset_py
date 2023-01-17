# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import mxnet as  mx

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import logging

logging.basicConfig(level=logging.DEBUG)
data=pd.read_csv('../input/DJIA_table.csv')
def judge_label(x):

    if x >0.01:

        return 1

    else:

        return 0
data['Date']=pd.to_datetime(data['Date'])

data['Close_l1']=data['Close'].shift(-1)

data['High_l1']=data['High'].shift(-1)

data['Low_l1']=data['Low'].shift(-1)

data['trend_Close']=np.log(data['Close']/data['Close_l1'])

data['trend_Low']=np.log(data['Low']/data['Low_l1'])

data['trend_High']=np.log(data['High']/data['High_l1'])

data['label']=data['trend_Close'].shift(1)



data['label']=data['label'].apply(judge_label)
name_list=[]

for k in range(20):

    for c in ['trend_Close','trend_Low','trend_High']:

        name=c+'_l'+str(int(k))

        name_list.append(name)

        data[name]=data[c].shift(k)
X=data[19:-2][name_list].values

y=data[19:-2]['label'].values
from sklearn.cross_validation import train_test_split
y.shape
X_train, X_test, y_train, y_test = train_test_split(nX,y)
train_set=mx.io.NDArrayIter(data=nX,label=y,batch_size=30)
nX=X.reshape([len(X),3,20,1])
data=mx.symbol.Variable('data')

label=mx.symbol.Variable('label')

c1=mx.symbol.Convolution(data=data,kernel=(3,1),num_filter=16)

a1=mx.symbol.Activation(data=c1,act_type='relu')

f2=mx.symbol.Flatten(data=a1)

f3=mx.symbol.FullyConnected(data=f2,num_hidden=32)

model=mx.symbol.LogisticRegressionOutput(data=f3,label=label)

MODEL=mx.model.FeedForward(model,num_epoch=20,learning_rate=0.01)
help(MODEL.fit(X=nX,batch_end_callback=Speedometer(batch_size,30)))
help(mx.model.FeedForward)
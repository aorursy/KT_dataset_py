import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense,Dropout

from sklearn.preprocessing import MinMaxScaler,StandardScaler

from keras.optimizers import Adam
train=pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",sep=",",header=0)

test=pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",sep=",")



train.replace({'?':np.nan},inplace=True)

test.replace({'?':np.nan},inplace=True)



train.drop(['ID'],axis=1,inplace=True)

test.drop(['ID'],axis=1,inplace=True)



train=pd.get_dummies(columns=['Size'],data=train)

test=pd.get_dummies(columns=['Size'],data=test)



train.dropna(inplace=True)
x_train=train.drop(['Class'],axis=1)

y_train=train['Class']

x_test=test
SS=StandardScaler()

data=SS.fit_transform(x_train)

x_train=pd.DataFrame(data,columns=x_train.columns)

test_d=SS.transform(x_test)

x_test=pd.DataFrame(test_d,columns=x_test.columns)

x_train.shape
model=Sequential()

model.add(Dense(64,input_dim=13,activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(32,activation='tanh'))

model.add(Dropout(0.2))

model.add(Dense(6,activation='softmax'))



model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'],optimizer='Adam')
model.summary()

model.fit(x_train,y_train,validation_split=0.1,epochs=10,batch_size=32)
y_pred=model.predict(x_test)

y_test=[]

for i,x in enumerate(y_pred):

    y_test.append(np.argmax(x))
x=pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",sep=",")
ans_df=pd.DataFrame({'ID':np.array(x['ID']),'Class':y_test})

ans_df['Class'].unique()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(ans_df)
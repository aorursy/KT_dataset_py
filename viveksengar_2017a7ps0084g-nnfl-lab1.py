import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')



from keras.layers import Dense, Dropout

from keras.models import Sequential



from sklearn.metrics import mean_absolute_error
df = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv',na_values =['?'])
df.fillna(df.mode(),inplace= True)

df.isnull().sum()

df.dropna(inplace = True)
df=pd.get_dummies(df,columns=['Size'], prefix=['Siz'])
df.info()
y=df['Class']
df.drop(['Class','ID'],axis=1,inplace=True)
X=df
df.head()
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import RobustScaler

scaler=RobustScaler()

X_train=scaler.fit_transform(X_train)

X_test=scaler.fit_transform(X_test)
from keras.utils import to_categorical

y_train=to_categorical(y_train,6)

y_test=to_categorical(y_test,6)
X_train.shape, y_test.shape, X_test.shape, y_test.shape
model=Sequential()

model.add(Dense(58,input_dim=13,activation='elu'))

model.add(Dropout(rate=0.1))

model.add(Dense(114,activation='elu'))

model.add(Dropout(rate=0.1))

model.add(Dense(32,activation='elu'))

#model.add(Dropout(rate=0.1))

model.add(Dense(6,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,verbose=1,validation_split=0.2,epochs=80,batch_size=20)
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=2),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
model.fit(X_train, y_train, validation_split=0.2, epochs=80,callbacks=callbacks,verbose=1)
test_results = model.evaluate(X_test, y_test, verbose=1)

print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
pred=model.predict_classes(X_test)
pred
y_test.argmax(1)
np.unique(pred,return_counts=True)
np.unique(y_test.argmax(1),return_counts=True)
df1=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df1.head()
df1=pd.get_dummies(df1,columns=['Size'],prefix=['Siz'])
X1=df1
X1.drop('ID',axis=1,inplace=True)
X1.head()
X1=scaler.fit_transform(X1)
fin1=model.predict_classes(X1)
np.unique(fin1,return_counts=True)
df2=pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df3=pd.DataFrame(index=df2['ID'])

df3['Class']=fin1
df3.to_csv('out.csv')
dff=pd.read_csv('/kaggle/working/out.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(dff, title = "Download CSV file", filename = "data.csv"):

    csv = dff.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(dff)
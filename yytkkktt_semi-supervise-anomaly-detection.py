import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split
%matplotlib inline
df1=pd.read_csv('../input/creditcard.csv')
df1.head()
df=df1.drop(columns=['Time','Amount'])
df.head()
df.describe()
df_normal=df[df['Class']==0]
df_normal.shape
df_anomaly=df[df['Class']==1]
df_anomaly.shape
df_anomaly1,df_anomaly2=train_test_split(df_anomaly,test_size=0.5)
df_anomaly1.shape
df_train,df_v=train_test_split(df_normal,test_size=0.005)
df_v.shape
df_v1,df_t1=train_test_split(df_v,test_size=0.5)
df_v1.shape
df_val=df_v1.append(df_anomaly1).sample(frac=1)
df_val.head()
df_test=df_t1.append(df_anomaly2).sample(frac=1)
df_test.head()
print(df_train.shape)
print(df_val.shape)
print(df_test.shape)
X_train=df_train.iloc[:,:-1].values
X_val=df_val.iloc[:,:-1].values
y_val=df_val.iloc[:,-1].values
X_test=df_test.iloc[:,:-1].values
y_test=df_test.iloc[:,-1].values
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_val=sc.transform(X_val)
X_test=sc.transform(X_test)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
inputshape=X_train[0].shape
num_train_sample=len(X_train)
num_val_sample=len(X_val)
batchsize=64
from keras.layers import Dense,Dropout,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
input1=Input(shape=inputshape)
x1=Dense(16,activation='relu')(input1)
x2=Dense(8,activation='relu')(x1)
encoded=Dense(4,activation='relu')(x2)
d1=Dense(8,activation='relu')(encoded)
d2=Dense(16,activation='relu')(d1)
decoded=Dense(28,activation='sigmoid')(d2)
autoencoder=Model(inputs=input1,outputs=decoded)
autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history = autoencoder.fit(X_train, X_train,
                    epochs=10,
                    batch_size=batchsize,
                    shuffle=True,
                    validation_data=(X_val, X_val),
                    verbose=1)
X_val_pred=autoencoder.predict(X_val)
mse = np.mean(np.power(X_val - X_val_pred, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse*1000,'true_class': y_val})
error_df.describe()
threshholds=np.linspace(0,100,300)
f1score=[]
for t in threshholds:
    y_hat=error_df.reconstruction_error>t
    f1score.append(f1_score(y_val,y_hat))
f1score
scores = np.array(f1score)
scores.max()
scores.argmax()
threshholds[scores.argmax()]
final_thresh=threshholds[scores.argmax()]
X_test_pred=autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)
test_error_df = pd.DataFrame({'reconstruction_error': mse*1000,'true_class': y_test})
test_error_df['y_hat']=test_error_df.reconstruction_error>final_thresh
test_error_df.head()
f1_score(y_test,test_error_df.y_hat)
precision_score(y_test,test_error_df.y_hat)
recall_score(y_test,test_error_df.y_hat)
confusion_matrix(y_test,test_error_df.y_hat)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(X_train)
print(gmm.score(X_val))

y_scores=gmm.score_samples(X_val)
df_erroe_val = pd.DataFrame({'log_prob': y_scores,'true_class': y_val})
df_erroe_val.describe()
threshhold2=np.linspace(-400,96,500)
f1score_2=[]
for t in threshhold2:
    y_hat=df_erroe_val.log_prob<t
    f1score_2.append(f1_score(y_val,y_hat))
f1score_2=np.array(f1score_2)
f1score_2.max()
f1score_2.argmax()
final_thresh2=threshhold2[f1score_2.argmax()]
final_thresh2
y_scores_test=gmm.score_samples(X_test)
y_pred2=y_scores_test<final_thresh2
print(f1_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
confusion_matrix(y_test,y_pred2)
df.head()
X1=df.iloc[:,:-1].values
y1=df.iloc[:,-1].values
#X_train3,X_test3,y_train3,y_test3=train_test_split(X1,y1,test_size=0.02)
from sklearn.preprocessing import MinMaxScaler
sc2=MinMaxScaler()
sc2.fit(X1)
X_train3=sc2.transform(X1)

X_train3.shape
inputshape=X_train3[0].shape
num_train_sample=len(X_train3)
batchsize=64
from keras.layers import Dense,Dropout,Input
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
input1=Input(shape=inputshape)
x1=Dense(16,activation='relu')(input1)
x2=Dense(8,activation='relu')(x1)
encoded=Dense(2,activation='relu')(x2)
d1=Dense(8,activation='relu')(encoded)
d2=Dense(16,activation='relu')(d1)
decoded=Dense(28,activation='sigmoid')(d2)
autoencoder=Model(inputs=input1,outputs=decoded)
autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
history = autoencoder.fit(X_train3, X_train3,
                    epochs=3,
                    batch_size=batchsize,
                    shuffle=True,
                    verbose=1)
y1.sum()
X_train_pred3=autoencoder.predict(X_train3)
mse3 = np.mean(np.power(X_train3 - X_train_pred3, 2), axis=1)
error_df3 = pd.DataFrame({'reconstruction_error': mse3*1000,'true_class': y1})
error_df3.describe()
y_pred3=error_df3.reconstruction_error>4.13
confusion_matrix(y1,y_pred3)
precision_score(y1,y_pred3)
recall_score(y1,y_pred3)
f1_score(y1,y_pred3)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(X_train3)
print(gmm.score(X_train3))
y_scores=gmm.score_samples(X_train3)
df_erroe_val = pd.DataFrame({'log_prob': y_scores,'true_class': y1})
df_erroe_val.describe()
y_pred3=df_erroe_val.log_prob<52.84
confusion_matrix(y1,y_pred3)
precision_score(y1,y_pred3)
recall_score(y1,y_pred3)
f1_score(y1,y_pred3)

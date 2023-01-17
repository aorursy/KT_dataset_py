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
# Special thanks to DATAI(https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization) and Sachin Sharma(https://www.kaggle.com/sachinsharma1123)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df
df.columns
df.isnull().sum()
#remove the uunamed column
df=df.drop(['Unnamed: 32'],axis=1)
ax = sns.countplot(df.diagnosis,label="Count") 
df.diagnosis.value_counts()
df.drop(["id"],axis=1).corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.drop(["id"],axis=1).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in list(df.columns):
    if df[i].dtype=='object':
        df[i]=le.fit_transform(df[i])
df # B:0, M:1
df.describe()
data = pd.melt(df.drop("id",axis=1),id_vars="diagnosis",var_name="features",value_name="value")
data
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
y=df['diagnosis']
x=df.drop(['diagnosis','id'],axis=1)
x
mean = x.mean(axis=0)
std = x.std(axis=0)
pd.concat([mean,std],axis=1)
x_norm=(x - mean)/std
df_norm = pd.concat([y,x_norm],axis=1)
df_norm
data_norm = pd.melt(df_norm,id_vars="diagnosis",var_name="features",value_name="value")
data_norm
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.violinplot(x="features", y="value", hue="diagnosis", data=data_norm, split=True, inner="quart")
# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_norm)
plt.figure(figsize=(30,10))
plt.xticks(rotation=90)
sns.boxplot(x="features", y="value", hue="diagnosis", data=data_norm)
# sns.stripplot(x="features", y="value",data=data_norm, jitter = True, color = "black")
plt.show()
# Then create models
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
x_norm_train,x_norm_test,y_norm_train,y_norm_test=train_test_split(x_norm,y,random_state=0,test_size=0.2) #y_train==y_norm_train
y_one_hot = pd.get_dummies(y)
x_norm_train,x_norm_test,y_norm_one_hot_train,y_norm_one_hot_test=train_test_split(x_norm,y_one_hot,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
list_1=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_s=knn.predict(x_test)
    scores=accuracy_score(y_test,pred_s)
    list_1.append(scores)
sns.barplot(x=list(range(1,21)),y=list_1)
print(max(list_1))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
pred_1=lr.predict(x_test)
score_1=accuracy_score(y_test,pred_1)
cmx = confusion_matrix(y_test,pred_1)
sns.heatmap(cmx,annot=True,fmt="d")
print("score_1 = ",score_1)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
pred_2=rfc.predict(x_test)
score_2=accuracy_score(y_test,pred_2)
cmx = confusion_matrix(y_test,pred_2)
sns.heatmap(cmx,annot=True,fmt="d")
print("score_2 = ",score_2)
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
pred_3=gbc.predict(x_test)
score_3=accuracy_score(y_test,pred_3)
cmx = confusion_matrix(y_test,pred_3)
sns.heatmap(cmx,annot=True,fmt="d")
print("score_3 = ",score_3)
# random forest classifier gives the best score among all
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(x.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.sigmoid),
    keras.layers.Dense(2)
  ])
 
  optimizer = tf.optimizers.RMSprop(0.001)
 
  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model
model = build_model()
model.summary()
# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
 
EPOCHS = 15
# Store training stats
history = model.fit(x_norm_train, y_norm_one_hot_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['accuracy']),
           label='accuracy')
  plt.plot(history.epoch, np.array(history.history['loss']),
           label='loss')
  plt.plot(history.epoch, np.array(history.history['val_accuracy']),
           label = 'Val accuracy')
  plt.plot(history.epoch, np.array(history.history['val_loss']),
           label='Val loss')

  plt.legend()
  plt.ylim([0,1])
 
plot_history(history)
pred_4 =model.predict(x_norm_test) 
score_4=accuracy_score(np.argmax(y_norm_one_hot_test.values,axis=1),np.argmax(pred_4,axis=1))
score_4

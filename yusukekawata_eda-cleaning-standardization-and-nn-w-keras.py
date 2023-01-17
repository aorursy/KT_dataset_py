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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

import tensorflow as tf

from tensorflow import keras
df=pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

df
df.isnull().sum()
df.describe()
ax = sns.countplot(df.Outcome,label="Count") 

df.Outcome.value_counts()
df.corr()
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data = pd.melt(df,id_vars="Outcome",var_name="features",value_name="value")

data
plt.figure(figsize=(30,10))

plt.xticks(rotation=90)

sns.violinplot(x="features", y="value", hue="Outcome", data=data,split=True, inner="quart")
y=df["Outcome"]

x=df.drop(["Outcome"],axis=1)

mean = x.mean(axis=0)

std = x.std(axis=0)

pd.concat([mean,std],axis=1)
x_norm=(x - mean)/std

df_norm = pd.concat([y,x_norm],axis=1)

df_norm
data_norm = pd.melt(df_norm,id_vars="Outcome",var_name="features",value_name="value")

data_norm
plt.figure(figsize=(30,10))

plt.xticks(rotation=90)

sns.violinplot(x="features", y="value", hue="Outcome", data=data_norm, split=True, inner="quart")

# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_norm)
plt.figure(figsize=(30,10))

plt.xticks(rotation=90)

sns.boxplot(x="features", y="value", hue="Outcome", data=data_norm)

# sns.stripplot(x="features", y="value",data=data_norm, jitter = True, color = "black")

plt.show()
# Then create models
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

y_one_hot = pd.get_dummies(y)

x_train,x_test,y_one_hot_train,y_one_hot_test=train_test_split(x,y_one_hot,random_state=0,test_size=0.2)

x_norm_train,x_norm_test,y_norm_train,y_norm_test=train_test_split(x_norm,y,random_state=0,test_size=0.2) #y_train==y_norm_train

x_norm_train,x_norm_test,y_norm_one_hot_train,y_norm_one_hot_test=train_test_split(x_norm,y_one_hot,random_state=0,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

list_1=[]

for i in range(1,21):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_norm_train,y_norm_train)

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

 

EPOCHS = 20
# Store training stats

history = model.fit(x_norm_train, y_norm_one_hot_train, epochs=EPOCHS,

                    validation_split=0.2, verbose=0,

                    callbacks=[PrintDot()])

plot_history(history)
pred_4 =model.predict(x_norm_test) 

cmx = confusion_matrix(np.argmax(y_norm_one_hot_test.values,axis=1),np.argmax(pred_4,axis=1))

sns.heatmap(cmx,annot=True,fmt="d")

score_4=accuracy_score(np.argmax(y_norm_one_hot_test.values,axis=1),np.argmax(pred_4,axis=1))

score_4
def build_model_2():

  model = keras.Sequential([

    keras.layers.Dense(64, activation=tf.nn.relu,

                       input_shape=(x.shape[1],)),

    keras.layers.Dense(64, activation=tf.nn.sigmoid),

    keras.layers.Dense(1,activation='sigmoid')

  ])

 

  optimizer = tf.optimizers.RMSprop(0.001)

 

  model.compile(loss='binary_crossentropy',

                optimizer=optimizer,

                metrics=['accuracy'])

  return model



model_2 = build_model_2()

model_2.summary()
# Store training stats

history_2 = model_2.fit(x_norm_train, y_norm_train, epochs=EPOCHS,

                    validation_split=0.2, verbose=0,

                    callbacks=[PrintDot()])

plot_history(history_2)
pred_5 =model_2.predict(x_norm_test) 

cmx = confusion_matrix(y_norm_test,pred_5>0.5)

sns.heatmap(cmx,annot=True,fmt="d")

score_5=accuracy_score(y_norm_test,pred_5>0.5)

score_5
# neural net is not as good as other models

# then clean the data and try again



(df==0).sum()
# Glucose,BloodPressure, SkinThickness, Insulin, BMI, should not to be 0, and seems to be miising data. Okay let's replace it with mean of them.

print("Glucose",df["Glucose"].mean(),df[df["Glucose"]!=0].Glucose.mean())

print("BloodPressure",df["BloodPressure"].mean(),df[df["BloodPressure"]!=0].BloodPressure.mean())

print("SkinThickness",df["SkinThickness"].mean(),df[df["SkinThickness"]!=0].SkinThickness.mean())

print("Insulin",df["Insulin"].mean(),df[df["Insulin"]!=0].Insulin.mean())

print("BMI",df["BMI"].mean(),df[df["BMI"]!=0].BMI.mean())
df_fill = df

df_fill["Glucose"] = df_fill["Glucose"].replace(0,df[df["Glucose"]!=0].Glucose.mean())

df_fill["BloodPressure"] = df_fill["BloodPressure"].replace(0,df[df["BloodPressure"]!=0].BloodPressure.mean())

df_fill["SkinThickness"] = df_fill["SkinThickness"].replace(0,df[df["SkinThickness"]!=0].SkinThickness.mean())

df_fill["Insulin"] = df_fill["Insulin"].replace(0,df[df["Insulin"]!=0].Insulin.mean())

df_fill["BMI"] = df_fill["BMI"].replace(0,df[df["BMI"]!=0].BMI.mean())
df_fill
x_fill=df_fill.drop(["Outcome"],axis=1)

mean_fill = x_fill.mean(axis=0)

std_fill = x_fill.std(axis=0)

pd.concat([mean_fill,std_fill],axis=1)
x_fill_norm=(x_fill - mean_fill)/std_fill

df_fill_norm = pd.concat([y,x_fill_norm],axis=1)

df_fill_norm
data_fill_norm = pd.melt(df_fill_norm,id_vars="Outcome",var_name="features",value_name="value")

data_fill_norm
plt.figure(figsize=(30,10))

plt.xticks(rotation=90)

sns.violinplot(x="features", y="value", hue="Outcome", data=data_fill_norm, split=True, inner="quart")

# sns.swarmplot(x="features", y="value", hue="diagnosis", data=data_norm)
x_fill_norm_train,x_fill_norm_test,y_train,y_test=train_test_split(x_fill_norm,y,random_state=0,test_size=0.2)

model_3 = build_model_2()

model_3.summary()

# Store training stats

history_3 = model_3.fit(x_fill_norm_train, y_train, epochs=EPOCHS,

                    validation_split=0.2, verbose=0,

                    callbacks=[PrintDot()])

plot_history(history_3)
pred_6 =model_3.predict(x_fill_norm_test) 

cmx = confusion_matrix(y_test,pred_6>0.5)

sns.heatmap(cmx,annot=True,fmt="d") 

score_6=accuracy_score(y_test,pred_6>0.5)

score_6
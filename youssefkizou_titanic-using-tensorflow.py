# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix





import matplotlib.pyplot as plt

import matplotlib as mpl

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')

df_train.head()  

df_train.describe()  
df_train.isnull().sum()
neg, pos = np.bincount(df_train['Survived'])

fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_train[df_train['Survived']==1]),

            len(df_train[df_train['Survived']==0])

        ], 

        name='Train Survived'

    ),

]





for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train Survived distribution',

    height=400,

    width=400

)

fig.show()



try:

    del df_train['Name'], df_train['Ticket'], df_train['Embarked'], df_train['Cabin'], df_train['PassengerId']

    del df_test['Name'], df_test['Ticket'], df_test['Embarked'], df_test['Cabin'], df_test['PassengerId']

    

except :

    pass



df_train.loc[df_train['Sex'] == 'male', 'Sex'] = 1

df_train.loc[df_train['Sex'] == 'female', 'Sex'] = 2

df_train['Sex'] = df_train['Sex'].astype(int)

df_test.loc[df_test['Sex'] == 'male', 'Sex'] = 1

df_test.loc[df_test['Sex'] == 'female', 'Sex'] = 2

df_test['Sex'] = df_test['Sex'].astype(int)



df_train['Age'] = df_train['Age'].fillna(30)

df_test['Age'] = df_test['Age'].fillna(30)



df_train.head()
f = plt.figure(figsize=(11, 13))

plt.matshow(df_train.corr(), fignum=f.number)

plt.xticks(range(df_train.shape[1]), df_train.columns, fontsize=14, rotation=75)

plt.yticks(range(df_train.shape[1]), df_train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
train_arr=df_train.values.tolist()

data=[x[1:] for x in train_arr]

response=[x[0] for x in train_arr]

data = np.array(data, dtype='float')

response = np.array(response, dtype='float')

# split into 60% for training and 640% for testing

from sklearn.model_selection import train_test_split



data_training, data_testing, response_training, response_testing = train_test_split(data, response, test_size=0.5, random_state=42)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

data_training = scaler.fit_transform(data_training)

data_testing = scaler.transform(data_testing)



data_training = np.clip(data_training, -5, 5)

data_testing = np.clip(data_testing, -5, 5)





print('Training labels shape:', response_training.shape)

print('Test labels shape:', response_testing.shape)



print('Training features shape:', data_training.shape)

print('Test features shape:', data_testing.shape)

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
METRICS = [

      tf.keras.metrics.BinaryAccuracy(name='accuracy'),

      tf.keras.metrics.Precision(name='precision'),

      tf.keras.metrics.Recall(name='recall'),

      tf.keras.metrics.AUC(name='auc'),

]



def make_model(metrics = METRICS, output_bias=None):

  if output_bias is not None:

    output_bias = tf.keras.initializers.Constant(output_bias)

  model = tf.keras.Sequential([

      tf.keras.layers.Dense(32, activation='relu'),

      tf.keras.layers.Dropout(0.5),

      tf.keras.layers.Dense(1, activation='sigmoid',

                         bias_initializer=output_bias),

  ])



  model.compile(

      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),

      loss=tf.keras.losses.BinaryCrossentropy(),

      metrics=metrics)



  return model
EPOCHS = 100



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)
model = make_model()

history = model.fit(

    data_training,

    response_training,

    epochs=EPOCHS,

    callbacks=[early_stopping],

    validation_data=(data_testing, response_testing))
def plot_metrics(history):

  metrics =  ['loss', 'auc', 'precision', 'accuracy']

  for n, metric in enumerate(metrics):

    name = metric.replace("_"," ").capitalize()

    plt.subplot(2,2,n+1)

    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')

    plt.plot(history.epoch, history.history['val_'+metric],

             color=colors[0], linestyle="--", label='Val')

    plt.xlabel('Epoch')

    plt.ylabel(name)

    if metric == 'loss':

      plt.ylim([0, plt.ylim()[1]])

    elif metric == 'auc':

      plt.ylim([0.8,1])

    else:

      plt.ylim([0,1])



    plt.legend()

mpl.rcParams['figure.figsize'] = (12, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



plot_metrics(history)
predict_train = model.predict_classes(data_training)

predict_test = model.predict_classes(data_testing)
cm = confusion_matrix(response_testing, predict_test)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt='g')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



unique, counts = np.unique(response_testing, return_counts=True)

print(dict(zip(unique, counts)))



unique, counts = np.unique(predict_test, return_counts=True)

print(dict(zip(unique, counts)))

weight_for_0 = (1 / neg)*(neg+pos)/2.0 

weight_for_1 = (1 / pos)*(neg+pos)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
weighted_model = make_model()



weighted_history = weighted_model.fit(

    data_training,

    response_training,

    epochs=EPOCHS,

    validation_data=(data_testing, response_testing),

    # The class weights go here

    class_weight=class_weight) 
plot_metrics(weighted_history)
predict_weighted_train = weighted_model.predict_classes(data_training)

predict_weighted_test = weighted_model.predict_classes(data_testing)
cm = confusion_matrix(response_testing, predict_weighted_test)



ax= plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, fmt='g')



ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')



unique, counts = np.unique(response_testing, return_counts=True)

print(dict(zip(unique, counts)))



unique, counts = np.unique(predict_weighted_test, return_counts=True)

print(dict(zip(unique, counts)))
data_test=df_test.values.tolist()

data_test = np.array(data_test, dtype='float')

print(data_test[0:4])
prediction = model.predict_classes(data_test)



id=['PassengerId']

for i in data_test:

    id.append(int(i[0]))



s=['Survived']

result = np.append(s, prediction[:, 0])





combined=np.vstack((id, result)).T

print(combined[0:4])
np.savetxt('Submission.csv', combined, delimiter=',', fmt='%s')
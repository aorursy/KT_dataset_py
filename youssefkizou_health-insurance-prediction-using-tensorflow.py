import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

import os

import tempfile

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



from imblearn.over_sampling import RandomOverSampler,SMOTE

from imblearn.under_sampling  import RandomUnderSampler



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go
mpl.rcParams['figure.figsize'] = (12, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
df_train=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

df_test=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

df_train.head()  
df_train.isnull().sum()
neg, pos = np.bincount(df_train['Response'])

fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_train[df_train['Response']==1]),

            len(df_train[df_train['Response']==0])

        ], 

        name='Train Response'

    ),

]





for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train Response distribution',

    height=400,

    width=400

)

fig.show()



df_train.loc[df_train['Gender'] == 'Male', 'Gender'] = 1

df_train.loc[df_train['Gender'] == 'Female', 'Gender'] = 2

df_train['Gender'] = df_train['Gender'].astype(int)

df_test.loc[df_test['Gender'] == 'Male', 'Gender'] = 1

df_test.loc[df_test['Gender'] == 'Female', 'Gender'] = 2

df_test['Gender'] = df_test['Gender'].astype(int)





df_train.loc[df_train['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2

df_train.loc[df_train['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1

df_train.loc[df_train['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0

df_train['Vehicle_Age'] = df_train['Vehicle_Age'].astype(int)

df_test.loc[df_test['Vehicle_Age'] == '> 2 Years', 'Vehicle_Age'] = 2

df_test.loc[df_test['Vehicle_Age'] == '1-2 Year', 'Vehicle_Age'] = 1

df_test.loc[df_test['Vehicle_Age'] == '< 1 Year', 'Vehicle_Age'] = 0

df_test['Vehicle_Age'] = df_test['Vehicle_Age'].astype(int)





df_train.loc[df_train['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1

df_train.loc[df_train['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

df_train['Vehicle_Damage'] = df_train['Vehicle_Damage'].astype(int)

df_test.loc[df_test['Vehicle_Damage'] == 'Yes', 'Vehicle_Damage'] = 1

df_test.loc[df_test['Vehicle_Damage'] == 'No', 'Vehicle_Damage'] = 0

df_test['Vehicle_Damage'] = df_test['Vehicle_Damage'].astype(int)

df_train.head()
f = plt.figure(figsize=(11, 13))

plt.matshow(df_train.corr(), fignum=f.number)

plt.xticks(range(df_train.shape[1]), df_train.columns, fontsize=14, rotation=75)

plt.yticks(range(df_train.shape[1]), df_train.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Male', 'Female'], 

        y=[

            len(df_train[df_train['Gender']==1]),

            len(df_train[df_train['Gender']==2])

        ], 

        name='Train Gender',

        text = [

            str(round(100 * len(df_train[df_train['Gender']==1]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Gender']==2]) / len(df_train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Male', 'Female'], 

        y=[

            len(df_test[df_test['Gender']==1]),

            len(df_test[df_test['Gender']==2])

        ], 

        name='Test Gender',

        text=[

            str(round(100 * len(df_test[df_test['Gender']==1]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Gender']==2]) / len(df_test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test gender column',

    height=400,

    width=700

)

fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_train[df_train['Driving_License']==1]),

            len(df_train[df_train['Driving_License']==0])

        ], 

        name='Train Driving_License',

        text = [

            str(round(100 * len(df_train[df_train['Driving_License']==1]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Driving_License']==0]) / len(df_train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_test[df_test['Driving_License']==1]),

            len(df_test[df_test['Driving_License']==0])

        ], 

        name='Test Driving_License',

        text=[

            str(round(100 * len(df_test[df_test['Driving_License']==1]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Driving_License']==0]) / len(df_test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Driving_License column',

    title_x=0.5,

    height=400,

    width=700

)

fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_train[df_train['Previously_Insured']==1]),

            len(df_train[df_train['Previously_Insured']==0])

        ], 

        name='Train Previously_Insured',

        text = [

            str(round(100 * len(df_train[df_train['Previously_Insured']==1]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Previously_Insured']==0]) / len(df_train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_test[df_test['Previously_Insured']==1]),

            len(df_test[df_test['Previously_Insured']==0])

        ], 

        name='Test Previously_Insured',

        text = [

            str(round(100 * len(df_test[df_test['Previously_Insured']==1]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Previously_Insured']==0]) / len(df_test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Previously_Insured column',

    title_x=0.5,

    height=400,

    width=700

)

fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_train[df_train['Vehicle_Damage']==1]),

            len(df_train[df_train['Vehicle_Damage']==0])

        ], 

        name='Train Vehicle_Damage',

        text = [

            str(round(100 * len(df_train[df_train['Vehicle_Damage']==1]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Vehicle_Damage']==0]) / len(df_train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['Yes', 'No'], 

        y=[

            len(df_test[df_test['Vehicle_Damage']==1]),

            len(df_test[df_test['Vehicle_Damage']==0])

        ], 

        name='Test Vehicle_Damage',

        text = [

            str(round(100 * len(df_test[df_test['Vehicle_Damage']==1]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Vehicle_Damage']==0]) / len(df_test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Vehicle_Damage column',

    title_x=0.5,

    height=400,

    width=700

)

fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Bar(

        x=['> 2 Years', '1-2 Year', '< 1 Year'], 

        y=[

            len(df_train[df_train['Vehicle_Age']==2]),

            len(df_train[df_train['Vehicle_Age']==1]),

            len(df_train[df_train['Vehicle_Age']==0])

        ], 

        name='Train Vehicle_Age',

        text = [

            str(round(100 * len(df_train[df_train['Vehicle_Age']==2]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Vehicle_Age']==1]) / len(df_train), 2)) + '%',

            str(round(100 * len(df_train[df_train['Vehicle_Age']==0]) / len(df_train), 2)) + '%'

        ],

        textposition='auto'

    ),

    go.Bar(

        x=['> 2 Years', '1-2 Year', '< 1 Year'], 

        y=[

            len(df_test[df_test['Vehicle_Age']==2]),

            len(df_test[df_test['Vehicle_Age']==1]),

            len(df_test[df_test['Vehicle_Age']==0])

        ], 

        name='Test Vehicle_Age',

        text = [

            str(round(100 * len(df_test[df_test['Vehicle_Age']==2]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Vehicle_Age']==1]) / len(df_test), 2)) + '%',

            str(round(100 * len(df_test[df_test['Vehicle_Age']==0]) / len(df_test), 2)) + '%'

        ],

        textposition='auto'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Vehicle_Age column',

    title_x=0.5,

    height=400,

    width=700

)

fig.show()
fig = make_subplots(rows=1, cols=2)



traces = [

    go.Histogram(

        x=df_train['Age'], 

        name='Train Age'

    ),

    go.Histogram(

        x=df_test['Age'], 

        name='Test Age'

    ),



]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 2) + 1, (i % 2)  +1)



fig.update_layout(

    title_text='Train/test Age column distribution',

    title_x=0.5,

    height=500,

    width=900

)

fig.show()
train_arr=df_train.values.tolist()

data=[x[:-1] for x in train_arr]

response=[x[-1] for x in train_arr]

data = np.array(data, dtype='float')

response = np.array(response, dtype='float')

# split into 40% for training and 60% for testing

data_training, data_testing, response_training, response_testing = train_test_split(data, response, test_size=0.4, random_state=42)

bool_response_training = response_training != 0

scaler = StandardScaler()

data_training = scaler.fit_transform(data_training)

data_testing = scaler.transform(data_testing)



data_training = np.clip(data_training, -5, 5)

data_testing = np.clip(data_testing, -5, 5)





print('Training labels shape:', response_training.shape)

print('Test labels shape:', response_testing.shape)



print('Training features shape:', data_training.shape)

print('Test features shape:', data_testing.shape)
METRICS = [

      tf.keras.metrics.TruePositives(name='tp'),

      tf.keras.metrics.FalsePositives(name='fp'),

      tf.keras.metrics.TrueNegatives(name='tn'),

      tf.keras.metrics.FalseNegatives(name='fn'), 

      tf.keras.metrics.BinaryAccuracy(name='accuracy'),

      tf.keras.metrics.Precision(name='precision'),

      tf.keras.metrics.Recall(name='recall'),

      tf.keras.metrics.AUC(name='auc'),

]



def make_model(metrics = METRICS, output_bias=None):

  if output_bias is not None:

    output_bias = tf.keras.initializers.Constant(output_bias)

  model = tf.keras.Sequential([

      tf.keras.layers.Dense(16, activation='relu'),

      tf.keras.layers.Dense(1, activation='sigmoid',

                         bias_initializer=output_bias),

  ])



  model.compile(

      optimizer=tf.keras.optimizers.Adam(lr=1e-3),

      loss=tf.keras.losses.BinaryCrossentropy(),

      metrics=metrics)



  return model

EPOCHS = 100

BATCH_SIZE = 2000



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=10,

    mode='max',

    restore_best_weights=True)
model = make_model(output_bias  = np.log([pos/neg]))
model.predict(data_training[:10])
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')

model.save_weights(initial_weights)
history = model.fit(

    data_training,

    response_training,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(data_testing, response_testing))
def plot_metrics(history):

  metrics =  ['loss', 'auc', 'precision', 'recall']

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

def plot_roc(name, labels, predictions, **kwargs):

  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)



  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

  plt.xlabel('False positives [%]')

  plt.ylabel('True positives [%]')

  plt.grid(True)

  ax = plt.gca()

  ax.set_aspect('equal')
plot_roc("Train Baseline", response_training, predict_train, color=colors[0])

plot_roc("Test Baseline", response_testing, predict_test, color=colors[0], linestyle='--')

plt.legend(loc='lower right')
# Scaling by total/2 helps keep the loss to a similar magnitude.

# The sum of the weights of all examples stays the same.



weight_for_0 = (1 / neg)*(neg+pos)/2.0 

weight_for_1 = (1 / pos)*(neg+pos)/2.0



class_weight = {0: weight_for_0, 1: weight_for_1}



print('Weight for class 0: {:.2f}'.format(weight_for_0))

print('Weight for class 1: {:.2f}'.format(weight_for_1))
weighted_model = make_model()

weighted_model.load_weights(initial_weights)



weighted_history = weighted_model.fit(

    data_training,

    response_training,

    batch_size=BATCH_SIZE,

    epochs=EPOCHS,

    callbacks = [early_stopping],

    validation_data=(data_testing, response_testing),

    # The class weights go here

    class_weight=class_weight) 
def plot_metrics(weighted_history):

  metrics =  ['loss', 'auc', 'precision', 'recall']

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
plot_roc("Train Baseline", response_training, predict_train, color=colors[0])

plot_roc("Test Baseline", response_testing, predict_test, color=colors[0], linestyle='--')



plot_roc("Train Weighted", response_training, predict_weighted_train, color=colors[1])

plot_roc("Test Weighted", response_testing, predict_weighted_test, color=colors[1], linestyle='--')





plt.legend(loc='lower right')
pos_features = data_training[bool_response_training]

neg_features = data_training[~bool_response_training]



pos_labels = response_training[bool_response_training]

neg_labels = response_training[~bool_response_training]
ids = np.arange(len(pos_features))

choices = np.random.choice(ids, len(neg_features))



res_pos_features = pos_features[choices]

res_pos_labels = pos_labels[choices]



print(res_pos_features.shape)

print(pos_features.shape)
resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)

resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)



order = np.arange(len(resampled_labels))

np.random.shuffle(order)

resampled_features = resampled_features[order]

resampled_labels = resampled_labels[order]



resampled_features.shape
BUFFER_SIZE = 100000



def make_ds(features, labels):

  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()

  ds = ds.shuffle(BUFFER_SIZE).repeat()

  return ds



pos_ds = make_ds(pos_features, pos_labels)

neg_ds = make_ds(neg_features, neg_labels)
for features, label in pos_ds.take(1):

  print("Features:\n", features.numpy())

  print()

  print("Label: ", label.numpy())
resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)
for features, label in resampled_ds.take(1):

  print(label.numpy().mean())
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

resampled_steps_per_epoch
resampled_model = make_model()

resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.

resampled_model = make_model(output_bias  = 0)



output_layer = resampled_model.layers[-1] 



val_ds = tf.data.Dataset.from_tensor_slices((data_testing, response_testing)).cache()

val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 



resampled_history = resampled_model.fit(

    resampled_ds,

    epochs=EPOCHS,

    steps_per_epoch=resampled_steps_per_epoch,

    callbacks = [early_stopping],

    validation_data=val_ds)
plot_metrics(resampled_history )
train_predictions_resampled = resampled_model.predict(data_training, batch_size=BATCH_SIZE)

test_predictions_resampled = resampled_model.predict(data_testing, batch_size=BATCH_SIZE)
plot_roc("Train Baseline", response_training, predict_train, color=colors[0])

plot_roc("Test Baseline", response_testing, predict_test, color=colors[0], linestyle='--')



plot_roc("Train Weighted", response_training, predict_weighted_train, color=colors[1])

plot_roc("Test Weighted", response_testing, predict_weighted_test, color=colors[1], linestyle='--')



plot_roc("Train Resampled", response_training, train_predictions_resampled,  color=colors[2])

plot_roc("Test Resampled", response_testing, test_predictions_resampled,  color=colors[2], linestyle='--')

plt.legend(loc='lower right')
data_test=df_test.values.tolist()

data_test = np.array(data_test, dtype='float')
prediction = resampled_model.predict_classes(data_test)

id=[]

for i in data_test:

    id.append(i[0])



id=np.array(id, dtype='int')

result = prediction[:, 0]

combined=np.vstack((id, result)).T
pd.DataFrame(combined).to_csv('sample_submission.csv')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

np.random.seed(0)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mpimg



import tensorflow as tf

tf.random.set_seed(1)

from tensorflow.keras import layers

from tensorflow.keras import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Define our example directories and files

base_dir = '../input/chest-xray-pneumonia/chest_xray'



train_dir = os.path.join( base_dir, 'train')

validation_dir = os.path.join( base_dir, 'val')

test_dir = os.path.join( base_dir,'test')



train_NORMAL_dir = os.path.join(train_dir, 'NORMAL') 

train_PNEUMONIA_dir = os.path.join(train_dir, 'PNEUMONIA')

validation_NORMALs_dir = os.path.join(validation_dir, 'NORMAL')

validation_PNEUMONIA_dir = os.path.join(validation_dir, 'PNEUMONIA')

test_NORMAL_dir = os.path.join(test_dir, 'NORMAL')

test_PNEUMONIA_dir = os.path.join(test_dir, 'PNEUMONIA')





train_NORMAL_fnames = os.listdir(train_NORMAL_dir)

train_PNEUMONIA_fnames = os.listdir(train_PNEUMONIA_dir)

validation_NORMAL_fnames = os.listdir(validation_NORMALs_dir)

validation_PNEUMONIA_fnames = os.listdir(validation_PNEUMONIA_dir)



#ratio of training set

train_ratio = (len(train_PNEUMONIA_fnames) + len(train_NORMAL_fnames))/(len(train_PNEUMONIA_fnames) + len(train_NORMAL_fnames)+len(validation_NORMAL_fnames) + len(validation_PNEUMONIA_fnames))



print(f'NORMAL class in Training set = {len(train_NORMAL_fnames)} : {round(len(train_NORMAL_fnames)/(len(train_NORMAL_fnames)+len(train_PNEUMONIA_fnames)),3)*100}%')

print(f'PNEUMONIA class in Training set = {len(train_PNEUMONIA_fnames)} : {round(len(train_PNEUMONIA_fnames)/(len(train_NORMAL_fnames)+len(train_PNEUMONIA_fnames)),3)*100}%')

print(f'Training set : Validation set ratio = {round(train_ratio*100,1)}% : {round((1-train_ratio)*100,1)}%')
old_train_set = []

old_validation_set = []



for (dirpath, dirnames, filenames) in os.walk(train_dir):

    old_train_set += [os.path.join(dirpath, file) for file in filenames]

for (dirpath, dirnames, filenames) in os.walk(validation_dir):

    old_validation_set += [os.path.join(dirpath, file) for file in filenames]



full_train_set = old_train_set + old_validation_set #combine old training and validation set together for further splitting

full_train_set = pd.DataFrame({'abs_path' : full_train_set}) #put path into 

full_train_set.loc[full_train_set['abs_path'].str.contains('NORMAL'), 'Class'] = 'NORMAL'

full_train_set.loc[full_train_set['abs_path'].str.contains('PNEUMONIA'), 'Class'] = 'PNEUMONIA'

full_train_set.sample(5)
X = full_train_set['abs_path']

y = full_train_set['Class']





val_split = 0.2



X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = val_split,stratify=y,random_state=42)



train_set = pd.DataFrame({'abs_path':X_train,'Class':y_train})

validation_set = pd.DataFrame({'abs_path':X_val,'Class':y_val})
datagen = ImageDataGenerator(rescale=1./255,

                            width_shift_range=0.1,

                            height_shift_range=0.1,

                            shear_range=0.1,

                            zoom_range=0.1,)

test_datagen = ImageDataGenerator(rescale=1./255)



#Train Image Generator

train_generator=datagen.flow_from_dataframe(

dataframe=train_set,

x_col="abs_path",

y_col="Class",

batch_size=32,

seed=42,

shuffle=False,

class_mode="binary",

target_size=(150,150))



#Validation Image Generator

validation_generator=test_datagen.flow_from_dataframe(

dataframe=validation_set,

x_col="abs_path",

y_col="Class",

batch_size=32,

seed=42,

shuffle=False,

class_mode="binary",

target_size=(150,150))



#Test Image Generator

test_generator = test_datagen.flow_from_directory(

test_dir,

batch_size=32,

shuffle=False,    

class_mode="binary",

target_size = (150,150))
normal_count = 0

pneumonia_count = 0

for i in range(len(train_generator.labels)):

    if train_generator.labels[i] == 0:

        normal_count += 1

    else:

        pneumonia_count += 1

        

assert(normal_count+pneumonia_count==len(train_generator.labels))





normal_weight = pneumonia_count/normal_count

pneumonia_weight = 1



class_weight = {0:normal_weight,1:pneumonia_weight}

print(class_weight)
val_normal_count = 0

val_pneumonia_count = 0

for i in range(len(validation_generator.labels)):

    if validation_generator.labels[i] == 0:

        val_normal_count += 1

    else:

        val_pneumonia_count += 1

        

print(f'normal in train set = {normal_count}')

print(f'pneumonia in train set = {pneumonia_count}')

print(f'normal in val set = {val_normal_count}')

print(f'pneumonia in val set = {val_pneumonia_count}')
x_batch, y_batch = next(train_generator)

fig = plt.figure(figsize = (20,20))



for i in range(25):

    ax = plt.subplot(5,5,i+1)

    plt.imshow(x_batch[i])

    if y_batch[i] == 1:

        plt.title('PNEUMONIA')

    else:

        plt.title('NORMAL')

from tensorflow.keras.applications.vgg16 import VGG16



def create_model():

    

    pre_trained_model = VGG16(input_shape=(150,150,3),include_top=False)

    

    #freeze layers weight 

    for layer in pre_trained_model.layers:

        layer.trainable = False

    

    

    last_layer = pre_trained_model.get_layer('block5_pool')

    print('last layer output shape: ', last_layer.output_shape)

    last_output = last_layer.output

    

    

    # Flatten the output layer to 1 dimension

    x = layers.Flatten()(last_output)

    # Add a fully connected layer with 128 hidden units and ReLU activation

    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001))(x)

    # Add a dropout rate of 0.2

    x = layers.Dropout(0.2)(x)

    # Add a final sigmoid layer for classification

    x = layers.Dense(1, activation='sigmoid')(x)           

    

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

    

    

    model = Model(pre_trained_model.input, x) 

    

    model.compile(optimizer = tf.keras.optimizers.SGD(

    learning_rate=0.001, momentum=0.9),

                  loss = 'binary_crossentropy', 

                  metrics = METRICS)

    

    model.summary()



    return model
#Training hyperparameters

batch_size = 32

steps_per_epoch = len(train_generator.labels) // batch_size

validation_step = len(validation_generator.labels) // batch_size

epochs = 30



early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor='val_auc', 

    verbose=1,

    patience=15,

    mode='max',

    restore_best_weights=True)



baseline_model = create_model()

baseline_history = baseline_model.fit(train_generator,

        validation_data = validation_generator,

        steps_per_epoch = steps_per_epoch,

        epochs = epochs,

        validation_steps = validation_step,

        callbacks = [early_stopping]

                            );



plt.style.use('ggplot')

from sklearn.metrics import confusion_matrix, roc_curve

def plot_metrics(history):

    metrics =  ['loss', 'auc', 'precision', 'recall']

    fig = plt.figure(figsize=(10,10))

    fig.suptitle('Preliminary performance of model', fontsize=16, y=1.05)

    for n, metric in enumerate(metrics):

        name = metric.replace("_"," ").capitalize()

        plt.subplot(2,2,n+1)

        sns.lineplot(history.epoch,  history.history[metric], label='Train')

        sns.lineplot(history.epoch, history.history['val_'+metric] ,label='Val')

        plt.xlabel('Epoch')

        plt.ylabel(name)

        if metric == 'loss':

          plt.ylim([0, plt.ylim()[1]])

        elif metric == 'auc':

          plt.ylim([0.8,1])

        else:

          plt.ylim([0,1])

    

        plt.legend()

        

    fig.tight_layout(pad=1.0)



def plot_cm(labels, predictions, p=0.5):

    cm = confusion_matrix(labels, predictions > p)

    plt.figure(figsize=(5,5))

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title('Confusion matrix @{:.2f}'.format(p))

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')



    print('Normal Chest X-rays Detected (True Negatives): ', cm[0][0])

    print('Normal Chest X-rays Incorrectly Detected (False Positives): ', cm[0][1])

    print('Pneumonia Chest X-rays Missed (False Negatives): ', cm[1][0])

    print('Pneumonia Chest X-rays Detected (True Positives): ', cm[1][1])

    print('Total Pneumonia Chest X-rays : ', np.sum(cm[1]))

    

def plot_roc(name, labels, predictions,zoom=False, **kwargs):

    fp, tp, _ = roc_curve(labels, predictions)

    

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)

    plt.xlabel('False positives [%]')

    plt.ylabel('True positives [%]')

    if zoom == True:

        plt.xlim([-0.5,20])

        plt.ylim([80,100.5])

    else:

        plt.xlim([-0.5,100])

        plt.ylim([0,100.5])

    plt.grid(True)

    ax = plt.gca()

    ax.set_aspect('equal')
plot_metrics(baseline_history)
train_prediction_baseline = baseline_model.predict(train_generator, batch_size = batch_size)

test_prediction_baseline = baseline_model.predict(test_generator, batch_size = batch_size)
plot_cm(test_generator.labels,test_prediction_baseline)
plot_roc("Train Baseline", train_generator.labels,train_prediction_baseline)

plot_roc("Test Baseline", test_generator.labels,test_prediction_baseline, linestyle='--')

plt.legend(loc='lower right')
plot_roc("Train Baseline", train_generator.labels,train_prediction_baseline,zoom=True)

plot_roc("Test Baseline", test_generator.labels,test_prediction_baseline, linestyle='--',zoom=True)

plt.legend(loc='lower right')
#Checkpoint callback



checkpoint = tf.keras.callbacks.ModelCheckpoint("xray_model.h5",monitor='val_loss',save_best_only=True)



#Reduce LR on Plateau

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.1,

    patience=10,

    verbose=0,

    mode="auto",

    min_lr=0.0001)
finetuned_model = create_model()

#load weight from baseline model

finetuned_model.set_weights(baseline_model.get_weights());
#train some VGG16 layers

# for count,layer in enumerate(finetuned_model.layers):

#     if count in np.arange(15,23).tolist():

#         layer.trainable = True

        

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



    

finetuned_model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),

            loss = 'binary_crossentropy', 

            metrics = METRICS)

finetuned_model.summary()
epochs = 100

finetuned_history = finetuned_model.fit(train_generator,

        validation_data = validation_generator,

        steps_per_epoch = steps_per_epoch,

        epochs = epochs,

        validation_steps = validation_step,

        class_weight=class_weight, #introduce class weight into model training

        callbacks = [early_stopping,checkpoint,reduce_lr]

                            ) 
plot_metrics(finetuned_history)
train_prediction_finetuned = finetuned_model.predict(train_generator, batch_size = batch_size)

test_prediction_finetuned = finetuned_model.predict(test_generator, batch_size = batch_size)
plot_cm(test_generator.labels,test_prediction_finetuned)
plot_roc("Train baseline", train_generator.labels,train_prediction_baseline,color='r')

plot_roc("Test baseline", test_generator.labels,test_prediction_baseline, linestyle='--',color='r')

plot_roc("Train finetuned", train_generator.labels,train_prediction_finetuned,color='b')

plot_roc("Test finetuned", test_generator.labels,test_prediction_finetuned, linestyle='--',color='b')

plt.legend(loc='lower right')
plot_roc("Train baseline", train_generator.labels,train_prediction_baseline,color='r',zoom=True)

plot_roc("Test baseline", test_generator.labels,test_prediction_baseline, linestyle='--',color='r',zoom=True)

plot_roc("Train finetuned", train_generator.labels,train_prediction_finetuned,color='b',zoom=True)

plot_roc("Test finetuned", test_generator.labels,test_prediction_finetuned, linestyle='--',color='b',zoom=True)

plt.legend(loc='lower right')
finetuned_results = finetuned_model.evaluate(test_generator,batch_size=batch_size, verbose=0)

for name, value in zip(finetuned_model.metrics_names, finetuned_results):

    print(name, ': ', value)
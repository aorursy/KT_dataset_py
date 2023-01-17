import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

df_train.head()  
train_label=np.array([img[0] for img in df_train.values])

train_img = np.array([img[1:] for img in df_train.values])

test_img=np.array([img for img in df_test.values])



print(train_img.shape, train_label.shape)

print(test_img.shape)

from collections import Counter



Counter(train_label)
print(train_img[0])
train_img = np.array([np.array_split(img, 28) for img in train_img])

test_img = np.array([np.array_split(img, 28) for img in test_img])

print(train_img.shape)



from sklearn.model_selection import train_test_split



data_training, data_testing, response_training, response_testing = train_test_split(train_img, train_label, test_size=0.1, random_state=42)
print(data_training.shape, response_training.shape)

print(data_testing.shape, response_testing.shape)
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
response_training = np.array(tf.keras.utils.to_categorical(response_training, 10))

response_testing = tf.keras.utils.to_categorical(response_testing, 10)
data_training = np.expand_dims(data_training, axis=-1)

data_testing = np.expand_dims(data_testing, axis=-1)

print(data_testing.shape)
train_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)



train_datagen.fit(data_training)

test_datagen.fit(data_testing) 
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

        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28,1)),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),

        tf.keras.layers.Dense(10, activation='softmax')

  ])



  model.compile(

      optimizer='rmsprop',

      loss='categorical_crossentropy',

      metrics=metrics)



  return model
model = make_model()

model.summary()
EPOCHS = 5
history = model.fit_generator(train_datagen.flow(data_training,response_training),

    epochs=EPOCHS,

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

    plt.legend()
import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.rcParams['figure.figsize'] = (12, 10)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']



plot_metrics(history)
test_img = np.expand_dims(test_img, axis=-1)
prediction = model.predict_classes(test_img)



id=['ImageId']

l=['Label']



for i in range(28000):

    id.append(i+1)

    l.append(prediction[i])





combined=np.vstack((id, l)).T

print(combined[0:4])
np.savetxt('Submission.csv', combined, delimiter=',', fmt='%s')
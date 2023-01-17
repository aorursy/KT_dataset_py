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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import keras
import tensorflow as tf
from tqdm import tqdm
from keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
labels = os.listdir('/kaggle/input/walk-or-run/walk_or_run_train/train')
labels
x_train = []

for i in labels:
  path = '/kaggle/input/walk-or-run/walk_or_run_train/train/' + i
  folder_data = os.listdir(path)

  for j in folder_data:
    img = load_img(path + '/' + j, target_size=(200,200))
    img = img_to_array(img)
    img = img / 255.0
    x_train.append(img)

x_train = np.array(x_train)
x_train.shape
y1 = pd.DataFrame( np.zeros(299, dtype=int) , columns = ['labels'])
y2 = pd.DataFrame( np.ones(301, dtype=int) , columns=['labels'] )
y_train = y1.append(y2, ignore_index=True)
y_train.shape
y_train['labels'].value_counts()
x_test = []

for i in labels:
  path = '/kaggle/input/walk-or-run/walk_or_run_test/test/' + i
  folder_data = os.listdir(path)

  for j in folder_data:
    img = load_img(path + '/' + j, target_size=(200,200))
    img = img_to_array(img)
    img = img / 255.0
    x_test.append(img)

x_test = np.array(x_test)
x_test.shape
y1 = pd.DataFrame( np.zeros(82, dtype=int) , columns = ['labels'])
y2 = pd.DataFrame( np.ones(59, dtype=int) , columns=['labels'] )
y_test = y1.append(y2, ignore_index=True)
y_test.shape
y_test['labels'].value_counts()
def imshow(a):
  plt.imshow(np.squeeze(a[np.random.randint(0, 599, size = 1, dtype=int)]))

def test_imshow(a):
  plt.imshow(np.squeeze(a[np.random.randint(0, 141, size = 1, dtype=int)]))
imshow(x_train)
test_imshow(x_test)
densenet = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape = (200,200,3))
densenet.summary()
for layer in densenet.layers:
  layer.trainable = False

densenet.summary()
x = tf.keras.layers.Flatten()(densenet.output)
prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x)
prediction
model = tf.keras.models.Model(inputs = densenet.input, outputs = prediction)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
history = model.fit(x_train,y_train, epochs = 5, validation_split=0.2, batch_size=20)
def accuracy_learning_curves(x):
  plt.plot(x.history['accuracy'])
  plt.plot(x.history['val_accuracy'])
  plt.title("ACCURACY CURVES")
  plt.legend(labels = ['Accuracy', 'Val_Accuracy'])
    
def loss_learning_curves(x):
  plt.plot(x.history['loss'])
  plt.plot(x.history['val_loss'])
  plt.title("LOSS CURVES")
  plt.legend(labels = ["Loss", 'Val_Loss'])
accuracy_learning_curves(history)
loss_learning_curves(history)
y_pred = model.predict(x_test)
y_pred
y_pred = np.round(y_pred)

cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('truth')
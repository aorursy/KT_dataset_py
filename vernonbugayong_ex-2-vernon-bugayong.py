import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report        
FILE_PATH = '/kaggle/input/utensils/utensils_train.csv'
df = pd.read_csv(FILE_PATH)
df.head()
y = df['Label'].values
y[:5]

df['Label'].value_counts()
df.isnull().any().describe()
y_encoder = OneHotEncoder(sparse=False)
y_encoded = y_encoder.fit_transform(y.reshape(-1, 1))
y_encoded[:5]
y_encoder.categories_
X = df.drop('Label', axis=1).values
X = X.reshape(-1, 28, 28, 1)
for i in range(10):
    plt.imshow(X[i].reshape(28, 28))
    plt.show()
    print('Label:', y[i])
### Model Training
###Used Average Pooling since first letter of first name starts with V
###Used Convolutional Layer with 4 x 4 filter since N = 3 (vowels in surname) + 1
###Used at least 2 dense layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow import keras 
from tensorflow.keras import regularizers

from keras.layers import Convolution2D, AvgPool2D
#create model
model = Sequential()

model.add(Conv2D(32, (4,4), activation='relu', input_shape=(28,28,1), kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(32, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(AvgPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(64, (4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(AvgPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile('adam', 'categorical_crossentropy')

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

MODEL_PATH = 'checkpoints/model_at_{epoch:02d}.mdl'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH)

model.fit(X, y_encoded, batch_size=20, epochs=100, validation_split=0.1)
#### Get the test data 

testdata = pd.read_csv("/kaggle/input/utensils/utensils_test.csv")

X_test = testdata.drop('Label', axis=1).values
Y_test = testdata['Label'].values

testdata['Label'].value_counts()
predictions = model.predict(X_test.reshape(-1,28,28,1))

roc_auc_score(Y_test, predictions, multi_class="ovr", average="macro")
for i in range(10):
    plt.imshow(X_test[i].reshape(28,28))
    plt.show()
    print('Prediction:', (predictions[i]))
predictions[:5]
predictions = np.argmax(predictions, axis=1)
predictions[:15]

print (Y_test[:15])
print("The accuracy of the model is", (accuracy_score(Y_test, predictions)*100), "%")
print(classification_report(Y_test, predictions, labels=[0, 1, 2]))
cf_matrix = confusion_matrix(Y_test, predictions)
print(cf_matrix)
import seaborn as sns
sns.heatmap(cf_matrix, annot=True)
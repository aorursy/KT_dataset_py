import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense 
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../input/digit-recognizer/train.csv')
train_df.head()
test_df = pd.read_csv('../input/digit-recognizer/test.csv')
test_df.head()
len(test_df.iloc[0])
sum(train_df.isna().sum())
sum(test_df.isna().sum())
### split into X and y

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

### check balancy of labels:
y_train.value_counts()
### Normalization : 0-255 => 0-1

X_train = X_train/255.0
test_df = test_df/255.0
X_train.values
### Reshape the images   (28, 28)  => (28, 28, 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)
test_df = test_df.values.reshape(-1, 28, 28, 1)
### Label encodeing (One hot encoders)

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=10)
### split into training and validation set

np.random.seed(42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
X_train.shape, X_val.shape, y_train.shape, y_val.shape
### Build the model

model = Sequential()
model.add(Conv2D(32, (5,5),
                 padding='same',
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
### Compile the model

optimizer = RMSprop()

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
### train the model

batch_size = 86
epochs = 30

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
                    verbose=2)
history.history.keys()
### Let's plot the loss and accuracy

plt.plot(np.arange(epochs), history.history['loss'], color='orange', label='Training loss')
plt.plot(np.arange(epochs), history.history['val_loss'], color='blue', label='Validation loss')
plt.title('Loss')
plt.xlabel('#Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(np.arange(epochs), history.history['accuracy'], color='orange', label='Training accuracy')
plt.plot(np.arange(epochs), history.history['val_accuracy'], color='blue', label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('#Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

### Let's make predictions
predictions = model.predict(test_df)
predictions[0]
pred_bool = (predictions>0.5)
pred_bool[0]
pred_bool = pred_bool.astype(int)
labels = ['zero', 'one', 'two', 'three', 'four',
          'five', 'six', 'seven', 'eight', 'nine']
labels[np.argmax(pred_bool[0])]
### Now let's visualize the prediction

print('prediction on test data set:')
print('')
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(test_df[i].reshape(28,28))
    plt.title(labels[np.argmax(pred_bool[i])])
plt.tight_layout(h_pad=0.5)
plt.show()
### make submission file
label_col = []
for i in range(len(pred_bool)):
    label_col.append(np.argmax(pred_bool[i]))

label_col[:10]


submission = pd.DataFrame()
submission['ImageId'] = np.arange(1, len(test_df)+1)
submission['Label'] = label_col
submission
submission.to_csv('submission.csv', index=False)

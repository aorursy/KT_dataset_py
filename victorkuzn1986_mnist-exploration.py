# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read and explore train dataset
train_df = pd.read_csv('../input/train.csv')
train_df.head()
# prepare the features for training

# normalize the image
def normalize(x):
    return (x/127.5)-1.0

# prepare train features and labels
def prepare_features_and_labels(df):
    labels = []
    features = []
    
    for index, row in df.iterrows():
        feature = np.array(row[1:].tolist())
        feature = normalize(feature)
        feature = np.reshape(feature, (28,28))
        
        features.append(np.array([feature]))
        
        label = np.zeros(10)
        label[row['label']] = 1.0
        labels.append(label)
        
        
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels
features, labels = prepare_features_and_labels(train_df)

features = features.reshape(features.shape[0], 1, 28, 28).astype('float32')
print (features.shape)
print (labels.shape)
print (features[0])
features = features.reshape(features.shape[0], 28, 28, 1)
print (features[0])
#train_X, train_y, dev_X, dev_y = train_test_split(features, labels, test_size=0.2)
# imports for keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten, Conv2D, BatchNormalization, GlobalAveragePooling2D
# build the CNN for classification
def build_simple_model(input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=3, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Conv2D(16, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Dropout(.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Dropout(.3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model
model = build_simple_model((28,28,1))
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# define the number of epochs
num_epochs = 100
# define the batch size
batch_size = 512
from keras.callbacks import ModelCheckpoint
def train_model(model, train_X, train_y, batch_size=32, epochs=3):
    checkpoint_callback = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit(train_X, train_y, batch_size=batch_size, validation_split=0.2, epochs=epochs, callbacks=[checkpoint_callback])

train_model(model, features, labels, batch_size, num_epochs)
# load the successful weights
model.load_weights('./model.h5')
# explore the test dataset
test_df = pd.read_csv('../input/test.csv')
test_df.head()
# explore the sample submission
sample_submission_df = pd.read_csv('../input/sample_submission.csv')
sample_submission_df.head()
# prepare test examples
def prepare_test_examples(df):
    features = []
    
    for index, row in df.iterrows():
        feature = np.array(row.tolist())
        feature = normalize(feature)
        feature = np.reshape(feature, (28,28))
        
        features.append(np.array([feature]))

    features = np.array(features)
    
    return features
test_examples = prepare_test_examples(test_df)
example = test_examples[0]
print (example.shape)

#example = example.reshape(example.shape[1], example.shape[2],example.shape[0])
#print (example.shape)
# do a prediction
def predict(model, img):
    softmax = model.predict(img.reshape(1, img.shape[1], img.shape[2], img.shape[0]))
    return np.argmax(softmax)
print (predict(model, example))
labels = []

for ex in test_examples:
    labels.append(predict(model, ex))
    
print (labels)
    
sample_submission_df['Label'] = labels
sample_submission_df.head()
sample_submission_df.to_csv('./viktork_submission.csv', index=False)

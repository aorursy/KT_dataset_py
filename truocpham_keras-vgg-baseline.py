import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras import layers
from keras.preprocessing import image
from keras.layers import Activation, Conv2D, Flatten, LSTM, Dense, Bidirectional, Input, Dropout, BatchNormalization, CuDNNLSTM, GRU, CuDNNGRU, Embedding, GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling2D, AveragePooling2D
from keras.models import Model

import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Sequential
from keras import optimizers

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
os.listdir("../input/")
train_df = pd.read_csv("../input/train.csv")
train_df.head()
def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0
    
    for fig in data['Image']:
        # Load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return X_train
def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder
X = prepareImages(train_df, train_df.shape[0], "train")
X /= 255
y, label_encoder = prepare_labels(train_df['Id'])
y.shape
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
# Build the model - VGG
def get_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation = "relu", input_shape = (100, 100, 3)))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (6, 6), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(Conv2D(64, (9, 9), activation = "relu"))
    model.add(Dropout(0.625))
    
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.625))
    model.add(Dense(64, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(y.shape[1], activation = 'softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.001, decay = 1e-06), metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])

    print(model.summary())

    return model
model = get_model()
history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
plt.plot(history.history['categorical_accuracy'])
plt.title('Model categorical accuracy')
plt.ylabel('categorical accuracy')
plt.xlabel('Epoch')
plt.show()
test = os.listdir("../input/test/")
print(len(test))
col = ['Image']
test_df = pd.DataFrame(test, columns=col)
test_df['Id'] = ''
X = prepareImages(test_df, test_df.shape[0], "test")
X /= 255
predictions = model.predict(np.array(X), verbose=1)
for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.head(10)
test_df.to_csv('submission.csv', index=False)
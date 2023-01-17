# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

from keras.utils import to_categorical

from keras.preprocessing import image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm
train = pd.read_csv('/kaggle/input/anokha-ai-adept/train.csv')

train.head()

train['label'].value_counts().plot.bar()
train_image = []

for i in tqdm(range(train.shape[0])):

    img = image.load_img('/kaggle/input/anokha-ai-adept/Train/'+train['filename'][i], target_size=(64,64,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
y=train['label'].values

import numpy as np

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)

class_weights = {0:1.4 ,

                1: 0.8,

                2:1}
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
model = Sequential()

model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

# from sklearn.metrics import f1_score

# import numpy as np

# from keras.callbacks import Callback

# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# class Metrics(Callback):

#     def on_train_begin(self, logs={}):

#         self.val_f1s = []

#         self.val_recalls = []

#         self.val_precisions = []



#     def on_epoch_end(self, epoch, logs={}):

#         val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()

#         val_targ = self.model.validation_data[1]

#         _val_f1 = f1_score(val_targ, val_predict)

#         _val_recall = recall_score(val_targ, val_predict)

#         _val_precision = precision_score(val_targ, val_predict)

#         self.val_f1s.append(_val_f1)

#         self.val_recalls.append(_val_recall)

#         self.val_precisions.append(_val_precision)

#         print(" — val_f1: %f — val_precision: %f — val_recall %f ") %(_val_f1, _val_precision, _val_recall)

#         return



# metrics = Metrics()

# print (metrics.val_f1s)



from keras import backend as K



def recall_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



def precision_m(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision



def f1_m(y_true, y_pred):

    precision = precision_m(y_true, y_pred)

    recall = recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc',f1_m,precision_m, recall_m])
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_f1_m', 

                                           # mode='min',

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]


from sklearn.utils import class_weight

model.fit(X_train, y_train,  epochs=45, validation_data=(X_test, y_test), batch_size=16, class_weight=class_weights)
test_file = pd.read_csv('/kaggle/input/anokha-ai-adept/test.csv')
test_image = []

for i in tqdm(range(test_file.shape[0])):

    img = image.load_img('/kaggle/input/anokha-ai-adept/Test/'+test_file['filename'][i], target_size=(64,64,1), grayscale=True)

    img = image.img_to_array(img)

    img = img/255

    test_image.append(img)

test = np.array(test_image)
prediction = model.predict_classes(test)
sample = pd.read_csv('/kaggle/input/anokha-ai-adept/SampleSolution.csv')

sample['label'] = prediction

sample.to_csv('sample1.csv', header=True, index=False)
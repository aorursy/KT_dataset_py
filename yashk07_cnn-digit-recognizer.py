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

train_data = pd.read_csv('../input/digit-recognizer/train.csv')
train_data
test_data = pd.read_csv('../input/digit-recognizer/test.csv')
train_data.shape
X = train_data.drop(['label'],axis=1)
X
y = train_data['label']
y
y.value_counts()
import seaborn as sns
sns.countplot(y)
X = X/255.0
train_data = train_data/255.0
from tensorflow.keras import utils
y = utils.to_categorical(y,10) #data,number of classes
y #now we have catagorical form of y
X = X.values.reshape(-1,28,28,1)
test_data = test_data.values.reshape(-1,28,28,1)
print("X Shape:",X.shape,"\n test Shape:",test_data.shape)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state=2)

X_train.shape
input_shape = X.shape[1:]
Y_train.shape
input_shape
from tensorflow.keras.models import Sequential #Keras is an api that sits on top of tensorflow. Moreover tensorflow is google's plateform to built as deep learning model!
from tensorflow.keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
cnn_model = Sequential()


cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
cnn_model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
cnn_model.add(Dropout(0.25))

cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation = "relu"))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(10, activation = "softmax"))
from tensorflow.keras.optimizers import RMSprop
rms = RMSprop
cnn_model.compile(loss= 'categorical_crossentropy',optimizer =rms(lr=0.001),metrics= ['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
epochs = 15
batch_size = 86

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

# Fit the model
history = cnn_model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size,callbacks=[learning_rate_reduction]
                            )

                              
history.history.keys()
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss Progress during Training')
plt.ylabel('Training and validation loss')
plt.xlabel('Number of epochs')
plt.legend(['Training Loss','Validation Loss'])
from sklearn.metrics import confusion_matrix
predicted_classes = cnn_model.predict_classes(X_val)
Y_true = np.argmax(Y_val,axis = 1) 
cm = confusion_matrix(Y_true,predicted_classes)
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True)
result = cnn_model.predict_classes(test_data)
result
submission = pd.DataFrame()
submission['ImageId'] = pd.Series(range(1,28001))
submission['Label'] = result
submission
submission.to_csv('Save2',index=False)
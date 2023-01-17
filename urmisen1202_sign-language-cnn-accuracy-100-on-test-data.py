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

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



sns.set(style='white', context='notebook', palette='deep')
#Load the train and test data from the dataset

df_train=pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

df_test=pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
df_train.info()
df_test.info()
df_train.head()
df_test.head()
Y_train = df_train["label"]



# Drop 'label' column

X_train = df_train.drop(labels = ["label"],axis = 1) 



# free some space

del df_train 



g = sns.countplot(Y_train)



Y_train.value_counts()
Y_test = df_test["label"]



# Drop 'label' column

X_test = df_test.drop(labels = ["label"],axis = 1) 



# free some space

del df_test 



g = sns.countplot(Y_test)



Y_test.value_counts()
# Checking the training data dose it has any null value or not

X_train.isnull().any().describe()
# Checking the testing data dose it has any null value or not



X_test.isnull().any().describe()
X_train = np.array(X_train, dtype='float32')

X_test = np.array(X_test, dtype='float32')

Y_train = np.array(Y_train, dtype='float32')

Y_test = np.array(Y_test, dtype='float32')
print(X_train.shape)

print(X_test.shape)
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

# -1 is used to identify the total amount of training and testing samples respectively



X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)
# Normalize the data

X_train = X_train / 255.0

X_test = X_test / 255.0
# Encode labels to vectors by using one hot encoding(ex : 6 -> [0,0,0,0,0,0,1,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 25)

Y_test = to_categorical(Y_test, num_classes = 25)

print(Y_train[0])
# split the training data into train and validate arrays (will be used later)



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)
# Visualization of a images 



image = X_train[119, :].reshape((28, 28))



plt.imshow(image)

plt.show()
print(X_train.shape)

print(X_test.shape)
model = Sequential()
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dense(25, activation = "softmax"))
model.summary()
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)



epochs = 30 

batch_size = 86
# With data augmentation to prevent overfitting (accuracy 0.99286)



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
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# evaluate the keras model

accuracy = model.evaluate(X_train, Y_train)

print(f'Train results - Loss: {accuracy[0]} - Accuracy: {accuracy[1]*100}%')
# evaluate the keras model

accuracy = model.evaluate(X_val, Y_val)

print(f'validation test results - Loss: {accuracy[0]} - Accuracy: {accuracy[1]*100}%')
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation test'], loc='upper left')

plt.show()
# Predict the values from the validation dataset

predict_val = model.predict(X_val)



# Convert predictions classes to one hot vectors 

y_val_pred=( np.argmax(predict_val,axis=1))



# Convert validation observations to one hot vectors

y_true = np.argmax(Y_val,axis = 1) 
# Performance evaluation of the model for validation set

results = confusion_matrix(y_true,y_val_pred) 

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_true,y_val_pred))

print ('Report : ')

print (classification_report(y_true,y_val_pred))
# evaluate the keras model

accuracy = model.evaluate(X_test, Y_test)

print(f'validation test results - Loss: {accuracy[0]} - Accuracy: {accuracy[1]*100}%')
# Predict the values from the testing dataset

predict_test = model.predict(X_test)



# Convert predictions classes to one hot vectors 

y_pred=( np.argmax(predict_test,axis=1))



# Convert testing observations to one hot vectors

y_test = np.argmax(Y_test,axis = 1) 
# Performance evaluation of the model for testing set

results = confusion_matrix(y_test,y_pred) 

print ('Confusion Matrix :')

print(results) 

print ('Accuracy Score :',accuracy_score(y_test,y_pred))

print ('Report : ')

print (classification_report(y_test,y_pred))
results = pd.Series(y_pred,name="Label")
submission = pd.concat([pd.Series(range(1,10001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
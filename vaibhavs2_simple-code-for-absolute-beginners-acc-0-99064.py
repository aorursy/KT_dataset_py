import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import cv2
import keras
import seaborn as sns
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
print(f'Image shape from DataFrame: {train.iloc[0].shape}')
X_train = np.array(train.drop('label',axis=1)).reshape(-1,28,28,1)
y_train = np.array(train['label'])

X_test = np.array(test).reshape(-1,28,28,1)

print(f'Reshaped image for covnet: {X_train[0].shape}')
from keras.utils.np_utils import to_categorical
y_train_cat = to_categorical(y_train,10)
X_train = X_train/X_train.max()
X_test = X_test/X_test.max()
# y_train_cat = y_train_cat.reshape(len(y_train_cat),10)
X_train.shape
# Creating a model 
from keras.models import Sequential
model = Sequential()

# Adding Convolution, Pooling layers and Flatten
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
model.add(Conv2D(input_shape=(28,28,1),kernel_size=(4,4),filters=32,activation='relu',kernel_initializer=keras.initializers.he_uniform))

# Adding a dropout layer to prevent overfitting
model.add(keras.layers.Dropout(0.2))


# Adding Pooling Layer
model.add(MaxPool2D(pool_size=(2,2),strides=2))

# Adding a network of neurons
model.add(Dense(128,activation='relu'))


# Adding a dropout layer to prevent overfitting
model.add(keras.layers.Dropout(0.5))

# Adding a network of neurons
model.add(Dense(64,activation='relu'))


# Adding a dropout layer to prevent overfitting
model.add(keras.layers.Dropout(0.5))

# Adding a network of neurons
model.add(Dense(32,activation='relu'))


# Adding a dropout layer to prevent overfitting
model.add(keras.layers.Dropout(0.5))

# Flattening the data from 2D to a single dimension array to be used by the algorithm
model.add(Flatten())

# Adding softmax classifer for multilabel classifier
model.add(Dense(10,activation='softmax'))

# Compiling the model to decide upon the loss function for error measurement, optimizer for correcting weights attached with input in the back propogation method and an accuracy parameter
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ['accuracy'])
# Training model with 10 epochs 
model.fit(X_train,y_train_cat,epochs=10)
# Storing the results of the test set in a variable
predictions = model.predict_classes(X_test)

# # Saving the results in a file and final submission
# submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(predictions,name='Label')],axis = 1)
# submission.to_csv("output.csv",index=False)
from kerastuner import HyperModel

from keras.models import Sequential

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
def build_model(hp):
    model = Sequential([
        keras.layers.Conv2D(
        filters=hp.Int('conv1filter',min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('kernelsize1', values = [3,4,5]),
        activation=hp.Choice('convactchoice1', values = ['relu','sigmoid']),
        input_shape=(28,28,1)),
        
        keras.layers.Conv2D(
        filters=hp.Int('conv2filter',min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('kernelsize2', values = [3,4]),
        activation=hp.Choice('convactchoice2', values = ['relu','sigmoid'])),
            
        keras.layers.Flatten(),
        
        keras.layers.Dense(
        units=hp.Int('dense_1_units',min_value=32, max_value=128, step=16),
        activation=hp.Choice('activationchoice1', values = ['relu','sigmoid'])),
        
        keras.layers.Dense(
        units=hp.Int('dense_2_units',min_value=32, max_value=128, step=16),
        activation=hp.Choice('activationchoice2', values = ['relu','sigmoid'])),
        
        keras.layers.Dense(10,activation = 'softmax')
    ])
    
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics = ['accuracy'])
    
    return model
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

tuner_search = RandomSearch(build_model, objective = 'val_accuracy', max_trials = 5)
tuner_search.search(X_train,y_train_cat,epochs=3,validation_split=0.1)
best_params_model = tuner_search.get_best_models(num_models=1)[0]
best_params_model.summary()
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
                                 rotation_range=10,
                                 width_shift_range=0.10,
                                 height_shift_range=0.10,
                                 zoom_range=0.10)

# datagen.fit(X_train)
result = best_params_model.fit_generator(datagen.flow(X_train,y_train_cat,batch_size=32),epochs=10)
# best_params_model.fit(X_train,y_train_cat,epochs=10,validation_split=0.1, initial_epoch=3)
sns.set_style('dark')
fig, ax = plt.subplots(1,1, figsize=(20,10))
ax.set_title('Accuracy per Epochs')
ax.plot(result.epoch, result.history['accuracy'])
plt.show()
# Storing the results of the test set in a variable
predictions = best_params_model.predict_classes(X_test)

# Saving the results in a file and final submission
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pd.Series(predictions,name='Label')],axis = 1)
submission.to_csv("kerastuner.csv",index=False)

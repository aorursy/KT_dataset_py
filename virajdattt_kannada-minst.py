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





import seaborn as sns

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing import image

from keras.optimizers import RMSprop,Adam

from keras.callbacks import EarlyStopping, LearningRateScheduler

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

Dig_Min = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")
def norm_reshape(df):

    try:

        df = df.drop('label', axis=1)

    except:

        print("No label column")

    df = df/255

    df=df.values.reshape(-1,28,28,1)

    return df



def plot_acc(history):

    import matplotlib.pyplot as plt

    history_dict = history.history

    acc_values = history_dict['accuracy'] 

    val_acc_values = history_dict['val_accuracy']

    acc = history_dict['accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')

    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()




#Dig_Min = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")



## 3. Brief Notes on findings moving on 



data.head()



data.shape



test.shape



data.columns[1:]



data['label'].value_counts().index



sns.barplot(data['label'].value_counts().index,data['label'].value_counts())







Y_TRAIN = data['label']



#X_TRAIN = data.drop('label', axis=1)





#X_TRAIN = X_TRAIN/255

#Y_TRAIN = Y_TRAIN/255



#X_TRAIN.head()



#X_TRAIN=X_TRAIN.values.reshape(-1,28,28,1)



X_TRAIN = norm_reshape(data)



Y_VAL = Dig_Min['label']

Y_VAL = to_categorical(Y_VAL)



X_VAL = norm_reshape(Dig_Min)









X_TRAIN



plt.imshow(X_TRAIN[4][:,:,0])

print(Y_TRAIN[4])





Y_TRAIN = to_categorical(Y_TRAIN)



X_TRAIN.shape



model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))





model.add(Conv2D(256, kernel_size=3, activation='relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))





#x_train, x_valid, y_train, y_valid = train_test_split(X_TRAIN, Y_TRAIN, test_size = 0.10, random_state=42)
train_datagen = ImageDataGenerator(rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)
optimizer = Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999)

model.compile(loss = 'categorical_crossentropy',optimizer = optimizer,metrics=['accuracy'])
#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
#history1 = model.fit(X_TRAIN, Y_TRAIN, epochs=20, validation_data=(X_VAL, Y_VAL), batch_size=128, verbose=1, callbacks=[annealer])
history1 = model.fit_generator(train_datagen.flow(X_TRAIN, Y_TRAIN, batch_size=128),

                              steps_per_epoch=200,

                              epochs=25,

                              validation_data=(X_VAL, Y_VAL),

                              validation_steps=70,

                              callbacks=[annealer],

                              verbose=1,)
model.summary()
plot_acc(history1)
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')









#test = norm_reshape(test)

test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)



y_pre=model.predict(test)     ##making prediction

y_pre=np.argmax(y_pre,axis=1) ##changing the prediction intro labels



sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)



sample_sub.head()
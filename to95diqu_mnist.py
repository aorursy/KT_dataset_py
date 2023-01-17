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
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import matplotlib.pyplot as plt
file = "/kaggle/input/digit-recognizer/train.csv"

test_file = "/kaggle/input/digit-recognizer/test.csv"

data = pd.read_csv(file)

test_data = pd.read_csv(test_file)
data.shape
X = np.array(data.iloc[:,1:])

Y = np.array(data.iloc[:,0])

X_test = np.array(test_data)

#split training set in train and validation sets

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.2, random_state = 42)

#reshape the data

X_train = X_train.reshape(33600,28,28,1)

X_val = X_val.reshape(8400,28,28,1)

X_test = X_test.reshape(28000,28,28,1)

#normalize data

X_train.astype('float32')

X_test.astype('float32')

X_val.astype('float32')

X_train = X_train/255

X_test =X_test/255

X_val =X_val/255

#one hot encoding

one_hot_train = to_categorical(Y_train, 10)

one_hot_val = to_categorical(Y_val, 10)
from keras.preprocessing.image import ImageDataGenerator



zoom = 0.1

shift = 0.1

rotation = 20



datagen = ImageDataGenerator(rotation_range=rotation,

                             width_shift_range=shift, 

                             height_shift_range=shift,

                             zoom_range = [1-zoom,1+zoom]

                             )

# fit parameters from data

datagen.fit(X_train)



aug_train_X = []

aug_train_Y = []

batch_size = 1000

num_augmented = 0

for X_batch, y_batch in datagen.flow(X_train, one_hot_train, batch_size=batch_size, shuffle=False):

    aug_train_X.append(X_batch)

    aug_train_Y.append(y_batch)

    num_augmented += batch_size

    if num_augmented == 5*X_train.shape[0]:

        break
#put data in initial shape

final_X_train = np.concatenate(aug_train_X)

final_Y_train = np.concatenate(aug_train_Y)

#plot a few images to see the effect

for i in range(0, 9):

    plt.subplot(330 + 1 + i)

    plt.imshow(final_X_train[500+i][:,:,0], cmap=plt.get_cmap('gray'))

    plt.title((np.argmax(final_Y_train[500+i])))

plt.show()
final_Y_train.shape
from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten, Dense, Dropout
model = Sequential() #define model

model.add(Conv2D(64, kernel_size = 3, padding ='same', activation='relu', input_shape=(28, 28,1)))

model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation='relu'))

model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size = 3,padding = 'same', activation='relu'))

model.add(Conv2D(256, kernel_size = 3,padding = 'same', activation='relu'))

model.add(Conv2D(512, kernel_size = 3,padding = 'same', activation='relu'))

model.add(Conv2D(512, kernel_size = 3,padding = 'same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(4096, activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(4096, activation = 'relu'))

model.add(Dropout(0.1))

model.add(Dense(1000, activation='relu'))

model.add(Dense(10, activation = 'softmax'))
model.summary()
#!pip install keras-adabound

#from keras_adabound import AdaBound

#optimizer = AdaBound(lr=1e-3, final_lr=0.1)
#compile the model

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

#train on data

model.fit(final_X_train,final_Y_train, epochs=10, batch_size=100)



from keras.metrics import categorical_accuracy

from sklearn.metrics import accuracy_score
#evaluate model on validaiton set

score = model.evaluate(X_val, one_hot_val, verbose=0)

print('validation loss:{}'.format(score[0]))

print('validation accuracy:{}'.format(score[1]))
#save weights

model.save_weights("model_alex_net_weights.h5")

print("Saved model to disk")
y_pred = model.predict(X_test)

y_out = np.argmax(y_pred, axis=1)

ImageID = np.arange(len(y_out))+1

output = pd.DataFrame([ImageID,y_out]).T

output.rename(columns = {0:'ImageId', 1:'Label'})

#Out

output.to_csv('model1_MNIST_pred.csv', header =  ['ImageId', 'Label' ], index = None) 
for i in range(0, 9):

    plt.subplot(330 + 1 + i)

    plt.imshow(X_test[i][:,:,0], cmap=plt.get_cmap('gray'))

    plt.title(np.argmax(y_pred[i]))

# show the plot

plt.show()
from IPython.display import HTML

import base64



def create( df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = f'<a target="\_blank">{title}</a>'

    return HTML(html)
link = create(output, filename = "output.csv" ) 
link
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='/kaggle/working/model_plot.png', show_shapes=True, show_layer_names=True)
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
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from tensorflow import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense,Flatten, Conv2D, Dropout, MaxPooling2D

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)
IMG_ROWS = 28

IMG_COLS = 28

NUM_CLASSES = 10

TEST_SIZE= 0.2

RANDOM_STATE = 13

#MODEL

NO_EPOCHS = 50

BATCH_SIZE = 128



IS_LOCAL = False

train_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

test_data = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
print("Train dataset has",train_data.shape[0]," rows and", train_data.shape[1]," columns")

print("Test dataset has",test_data.shape[0]," rows and", test_data.shape[1]," columns")
# Create a dictionary for each type of label 

labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",

          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}



def sample_images_data(data):

    sample_images = []

    sample_labels = []

    

    for k in labels.keys():

        

        samples = data[data['label'] == k].head(4)

        

        for j,s in enumerate(samples.values):

            img = np.array(samples.iloc[j,1:]).reshape(IMG_ROWS,IMG_COLS)

            sample_images.append(img)

            sample_images.append(samples.iloc[j,0])

            

        print("Total number of sample images to plot:",len(sample_images))

        return sample_images,sample_labels



train_sample_images,train_sample_labels = sample_images_data(train_data) 
def plot_sample_images(data_sample_images,data_sample_labels,cmap="Blues"):

    # Plot the sample images now

    f, ax = plt.subplots(2,4, figsize=(16,10))



    for i, img in enumerate(data_sample_images):

        ax[i//8, i%8].imshow(img, cmap=cmap)

        ax[i//8, i%8].axis('off')

        ax[i//8, i%8].set_title(labels[data_sample_labels[i]])

    plt.show()    

    

plot_sample_images(train_sample_images,train_sample_labels, "Blues")
def data_preprocessing(raw):

    out_y = keras.utils.to_categorical(raw.label, NUM_CLASSES)

    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
X,y = data_preprocessing(train_data)

X_test,y_test = data_preprocessing(test_data)
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=TEST_SIZE,random_state=13)
print('X train - ',X_train.shape[0],'rows and', X_train.shape[1:4],'Columns.')

print('X valid - ',X_val.shape[0],'rows and', X_val.shape[1:4],'Columns.')

print('X test - ',X_test.shape[0],'rows and', X_test.shape[1:4],'Columns.')
#model

model = Sequential()



# Add convolution 2D

model.add(Conv2D(32,kernel_size=(3,3),

                activation='relu',

                kernel_initializer='he_normal',

                input_shape=(IMG_ROWS,IMG_COLS,1)))



model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Dropout(0.25))

model.add(Conv2D(64,

                kernel_size=(3,3),

                activation='relu',))



model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))



model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(NUM_CLASSES,activation='softmax'))





model.compile(loss=keras.losses.categorical_crossentropy,

             optimizer='adam',

             metrics=['accuracy'])
model.summary()
plot_model(model,to_file='model.png')

SVG(model_to_dot(model).create(prog='dot',format='svg'))
train_model = model.fit(X_train,y_train,

                       batch_size=BATCH_SIZE,

                       epochs=NO_EPOCHS,

                       verbose=1,

                       validation_data=(X_val,y_val))
score = model.evaluate(X_test,y_test,verbose=0)

print('Test loss',score[0])

print('Test accuracy',score[1])

plt.plot(train_model.history['accuracy'])

plt.plot(train_model.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train','val'],loc='upper left')

plt.show()



plt.plot(train_model.history['loss'])

plt.plot(train_model.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','val'],loc='upper left')

plt.show()
y_predicted = model.predict_classes(X_test)

y_true = test_data.iloc[:,0]
target_names = ['Class {} ({}) :' .format(i,labels[i]) for i in range(NUM_CLASSES)]

print(classification_report(y_true,y_predicted,target_names=target_names))
import numpy as np

import pandas as pd

import os

from os import listdir

from PIL import Image

from numpy import array

from numpy import asarray

import seaborn as sns



training_img_list = list()

pathToTrainData='/kaggle/input/plant-seedlings-classification/train'



shape_sum = 0

class_name_numb = dict()

train_avg_shape = 80



for dirname, _, filenames in os.walk(pathToTrainData):

    for filename in filenames:

        img_data = Image.open(os.path.join(dirname, filename))

        

        resizedImage = img_data.resize((train_avg_shape, train_avg_shape))

        resizedImage = resizedImage.convert('RGB')

        resizedImage = asarray(resizedImage)/255

        

        class_label = dirname.split('/')[-1]

        training_img_list.append([resizedImage, class_label])

        shape_sum += np.max(img_data.size)

        class_name_numb[class_label] = len(class_name_numb)-1
from keras import Sequential

from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from sklearn.model_selection import KFold

from numpy import asarray

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix



np.random.seed(17)

kf = KFold(n_splits=5)

epochs = 20

batch_size = 32
from keras.applications.vgg19 import VGG19

vgg19_model = VGG19(weights='imagenet',include_top=False)

from keras.utils import plot_model

plot_model(vgg19_model, to_file='VGG19Original.png', show_shapes=True, show_layer_names=True)
vgg19_model.summary()
x=vgg19_model.output
from keras.layers import Dense,GlobalAveragePooling2D



x=GlobalAveragePooling2D()(x)

x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3

preds=Dense(len(class_name_numb), activation='softmax')(x) #final layer with softmax activation
from keras.models import Model

newModel=Model(inputs=vgg19_model.input,outputs=preds)

print("changed model layer count %d" %len(newModel.layers))

newModel.summary()
for layer in newModel.layers[:-5]:

    layer.trainable=False



newModel.summary()
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.mobilenet import preprocess_input



train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory(pathToTrainData, 

                                                  target_size=(80,80),

                                                  color_mode='rgb',

                                                  batch_size=32,

                                                  class_mode='categorical',

                                                  shuffle=True)
newModel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])



step_size_train=train_generator.n//train_generator.batch_size

history = newModel.fit_generator(generator=train_generator,

                   steps_per_epoch=step_size_train,

                   epochs=10)
print("History for cross validation fold 1")

plt.plot(history.history['accuracy'])

plt.plot(history.history['loss'])

plt.title('Model loss and accuracy')

plt.xlabel('Epoch')

plt.legend(['accuracy','loss'], loc='upper right')

plt.show()
pathToTestData ='/kaggle/input/plant-seedlings-classification/test'

test_img_list = list()



for dirname, _, filenames in os.walk(pathToTestData):

    for filename in filenames:

        img_data = Image.open(os.path.join(dirname, filename))

        

        resizedImage = img_data.resize((train_avg_shape, train_avg_shape))

        resizedImage = resizedImage.convert('RGB')

        resizedImage = asarray(resizedImage)/255



        test_img_list.append([resizedImage,filename])
X_test = np.zeros((len(test_img_list), train_avg_shape, train_avg_shape, 3), dtype='float32')



for i,img in enumerate(test_img_list):

    X_test[i] = test_img_list[i][0]
predictions = newModel.predict(X_test, batch_size=None, verbose=0, steps=None, 

                              callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False)

predictions=pd.DataFrame(predictions)
inverse_label_map = dict()

for k,v in train_generator.class_indices.items():

    inverse_label_map[v] = k
pred_label_num = predictions.idxmax(axis=1)

pred_label_num_new = list()



for x in pred_label_num:

    y = inverse_label_map[x]

    pred_label_num_new.append(y)



pred_label_num_new = pd.DataFrame(pred_label_num_new)

print(pred_label_num_new[0])
pred=pd.DataFrame()
testImages = pd.DataFrame(test_img_list) 

pred.insert(0,'file',testImages[1])

pred.insert(1,'species',pred_label_num_new[0])

pred.head()
pred.to_csv('predictionsVgg19.csv',index = None, header=True)
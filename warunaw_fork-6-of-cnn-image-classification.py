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



# testing_img_list = list()

# pathToTestData ='/kaggle/input/plant-seedlings-classification/test'

    

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

        

# train_avg_shape = int(np.round(shape_sum / len(training_img_list)))
print("Class names in traininig data set:")

for items in class_name_numb.items():

    print(items)
print("training data set size:%d" %len(training_img_list))

print("Reshaping dimention %d" %train_avg_shape)
import random

random.shuffle(training_img_list)
X = np.zeros((len(training_img_list), train_avg_shape, train_avg_shape, 3), dtype='float32')

Y = np.zeros((len(training_img_list)))



for i,img in enumerate(training_img_list):

    X[i] = training_img_list[i][0]

    Y[i] = class_name_numb[training_img_list[i][1]]
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
fold = 0

history_records=list()

conf_matrices=list()

scores=list()



for train_index, test_index in kf.split(X):

    fold += 1

    print("#"*50)

    print("Running on fold:%d" %fold)

    

    cnn_model = Sequential()



    # CNN Layer 1

    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(train_avg_shape,train_avg_shape,3)))

    cnn_model.add(MaxPooling2D((2, 2)))



    # CNN Layer 2

    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    

    # CNN Layer 3

    cnn_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))



    cnn_model.add(Flatten())

    cnn_model.add(Dense(350, activation='relu'))

    cnn_model.add(Dense(len(class_name_numb) , activation='softmax'))



    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



    X_train = X[train_index]

    y_train = Y[train_index]

    X_test = X[test_index]

    y_test = Y[test_index]

    history = cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split = 0.1)

    history_records.append(history)

    

    y_predicted = cnn_model.predict(X_test)

    f = pd.DataFrame(y_predicted)

    y_predicted = f.idxmax(axis=1).values

    cnf_matrix = confusion_matrix(y_test, y_predicted)

    

    conf_matrices.append(cnf_matrix)

    scores.append(cnn_model.evaluate(X_test, y_test))

    

print("Average score for model")

scores = pd.DataFrame(scores) 

print("test loss=%f test accuracy=%f" %(np.average(scores.iloc[:][0]), np.average(scores.iloc[:][1])))



print("History for cross validation fold 1")

plt.plot(history_records[0].history['val_loss'])

plt.plot(history_records[0].history['loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['validation','Train'], loc='upper right')

plt.show()



fig, ax = plt.subplots(1,figsize=(10,10))

ax = sns.heatmap(conf_matrices[0], ax=ax, cmap=plt.cm.YlGn, annot=True)

ax.set_xticklabels(class_name_numb.keys())

ax.set_yticklabels(class_name_numb.keys())

plt.title('Confusion Matrix')

plt.ylabel('True class')

plt.xlabel('Predicted class')

fig.savefig('Confusion matrix.png', dpi=300)

plt.yticks(rotation=45)

plt.xticks(rotation=45)

plt.show();
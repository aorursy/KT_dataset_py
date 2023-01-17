import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from tensorflow.keras.preprocessing import image

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization,Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam
df = pd.read_excel(r'../input/bdrw/BDRW_train_2/BDRW_train_2/labels.xls',sheet_name='Sheet1')
filenames=[]

label=[]

filenames.append('digit_0')

label.append('1')

take=False

for i in range(len(df[1])):

    filenames.append(df['digit_0'][i])

    label.append(str(df[1][i]))

print(len(filenames))

    
width=224

height=224

X=[]

path2='../input/bdrw/BDRW_train_2/BDRW_train_2/{}.jpg'

path1='/kaggle/input/bdrw/BDRW_train_1/BDRW_train_1/{}.jpg'

for i in tqdm(range(len(label))):

    try:

        impath=path2.format(filenames[i])

        img = image.load_img(impath,target_size=(width,height,3))

    except FileNotFoundError :

        impath=path1.format(filenames[i])

        img = image.load_img(impath,target_size=(width,height,3))

    img = image.img_to_array(img)

    img=img/255.0

    X.append(img)

    

print(len(X))
X= np.array(X)
Y=pd.DataFrame(data=label,columns =['Label'])



print(Y.head())
def encode_and_bind(original_dataframe, feature_to_encode):

    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])

    res = pd.concat([original_dataframe, dummies], axis=1)

    res = res.drop([feature_to_encode], axis=1)

    return(res)
Y = encode_and_bind(Y, 'Label')



print(Y.head())
Y = Y.to_numpy()
Y.shape
plt.imshow(X[1012])
Y[1012]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, test_size=0.15)
model = Sequential()

model.add(Conv2D(16,(2,2), activation='relu', input_shape=X_train[0].shape))

model.add(BatchNormalization())

model.add(MaxPool2D(5,5))





model.add(Conv2D(32,(10,10), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(5,5))

model.add(Dropout(0.2))





model.add(Conv2D(32,(5,5), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(3,3))

model.add(Dropout(0.2))











model.add(Flatten())



model.add(Dense(128,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.5))









model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
iteration=100

history = model.fit(X_train, Y_train, batch_size=10, epochs=iteration, validation_data=(X_test,Y_test))
def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
%matplotlib inline

from sklearn.metrics import confusion_matrix

import itertools

import matplotlib.pyplot as plt
y_pred=model.predict_classes(X_test,verbose=0)

y_test=np.argmax(Y_test, axis=1)
cm = confusion_matrix(y_test,y_pred)
def plot_confusion_matrix(cm, classes,

                        normalize=False,

                        title='Confusion matrix',

                        cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

            horizontalalignment="center",

            color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
import os

files =path='../input/testdigit/'

path='../input/testdigit/{}'

test = os.listdir(files)

testimg=[]

testAr=[]

for i in test:

  impath=path.format(i)

  img = image.load_img(impath,target_size=(width,height,3))

  

  testimg.append(img)

  img = image.img_to_array(img)

  img=img/255.0

  plt.imshow(img)

  testAr.append(img)

prediction=[]

predt=[]

for i in testAr:

    i=i.reshape(1,width,height,3)

    pred = model.predict_classes(i)

    predt.append(pred)

    prediction.append(pred)

rows = 1

cols = len(testimg)

axes=[]

fig=plt.figure()



for a in range(rows*cols):

    b = np.random.randint(7, size=(height,width))

    axes.append( fig.add_subplot(2, 5, a+1) )

    subplot_title=('pred: '+str(prediction[a][0]))

    axes[-1].set_title(subplot_title)  

    plt.imshow(testimg[a])

plt.show()
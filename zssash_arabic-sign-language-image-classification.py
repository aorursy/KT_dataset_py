!conda config --env --set always_yes true

!conda install -c conda-forge arabic_reshaper

!conda install -c conda-forge python-bidi 
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import random # Generate pseudo-random numbers

from random import randint



from sklearn.utils import shuffle # Shuffle arrays or sparse matrices in a consistent way

from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets

from sklearn.metrics import classification_report, confusion_matrix

import sklearn



import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec # Specifies the geometry of the grid that a subplot can be placed in.



import keras

from keras import models as Models

from keras import layers as Layers

from keras.preprocessing import image

from keras.models import Sequential,Model

from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization

from keras.layers import Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint,EarlyStopping

from keras import utils as Utils

from keras.utils import to_categorical # Converts a class vector (integers) to binary class matrix.



from keras.utils.vis_utils import model_to_dot



import seaborn as sns



# from IPython.display import SVG



import arabic_reshaper # Reconstruct Arabic sentences to be used in applications that don't support Arabic

from bidi.algorithm import get_display
# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
# global variables

Language = "Ar"

ImageClassMapping_path = "../input/Labels/ImagesClassPath.csv"

ClassLabels_path = "../input/Labels/ClassLabels.xlsx"

ImagesRoot_path = "../input/"



ModelFileName ='Model_255.h5'
# load 54k image path mapping

df_ImageClassPath = pd.read_csv(ImageClassMapping_path)

display(df_ImageClassPath.head())
# load Class Labels

df_Classes = pd.read_excel(ClassLabels_path)

display(df_Classes.head())
df_ImageClassPath.groupby("ClassId").size().describe()


ddata = {"samples destribution":df_ImageClassPath.groupby("ClassId").size()}

iindex = range(32)



ddataframe = pd.DataFrame(data=ddata, index= iindex)

ddataframe.plot.bar(stacked= True, rot= 15, title='samples destribution')

plt.show(block= True)
# Split 54K Images into 3 groups of Fixed Prediction, training and test

# the dataset is 32 class,split is maintaind as per class 

def SplitData(predictions,testsize):

    

    min = df_ImageClassPath.groupby("ClassId").size().min()

    print('{0} Samples per Class'.format(min))

    

    # empty dataframes with same column difinition

    df_TrainingSet = df_ImageClassPath[0:0].copy()

    df_TestSet = df_ImageClassPath[0:0].copy()

    df_PredSet = df_ImageClassPath[0:0].copy()



    # Create the sets by loop thru classes and append

    for index,row in df_Classes.iterrows():

        # make sure all class are same size 

        df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']].sample(min,random_state= 42)

        

#         df_FullSet = df_ImageClassPath[df_ImageClassPath['ClassId'] == row['ClassId']]

        

        df_PredSet = df_PredSet.append(df_FullSet.sample(n=predictions, random_state=1))

        df_FullSet = pd.merge(df_FullSet,df_PredSet, indicator=True, 

                              how='left').query('_merge=="left_only"').drop('_merge', axis=1)

        

        trainingSet, testSet = train_test_split(df_FullSet, test_size= testsize)        

        

        df_TrainingSet = df_TrainingSet.append(trainingSet)

        df_TestSet = df_TestSet.append(testSet)

    

    return df_TrainingSet,df_TestSet,df_PredSet

# retrive class Label (Arabic or English) using class id 

def get_classlabel(class_code,lang= 'Ar'):

    if lang== 'Ar':

        text_to_be_reshaped = df_Classes.loc[df_Classes['ClassId'] == class_code, 

                                             'ClassAr'].values[0]

        reshaped_text = arabic_reshaper.reshape(text_to_be_reshaped)

        return get_display(reshaped_text)

    elif lang== 'En':

        return df_Classes.loc[df_Classes['ClassId'] == class_code, 'Class'].values[0]

    
# prepare Images, and class Arrays

def getDataSet(setType,isDL): # 'Training' for Training dataset , 'Testing' for Testing data set

    imgs = []

    lbls = []

    df = pd.DataFrame(None)

    

    if setType =='Training':

        df = dtTraining.copy()

    elif setType=='Test':

        df = dtTest.copy()

    elif setType=='Prediction':

        df = dtPred.copy()



    for index,row in df.iterrows():

        lbls.append(row['ClassId'])

        try:

            imageFilePath = os.path.join(ImagesRoot_path, row['ImagePath'])

            img = image.load_img(imageFilePath, target_size=(64,64,1), 

                                 color_mode = "grayscale")

            img = image.img_to_array(img) # to array

            img = img/255 # Normalize

            if isDL == False:

                img = img.flatten() # for knn_classifier Model

            imgs.append(img)



        except Exception as e:

            print(e)

            

    shuffle(imgs,lbls,random_state=255) #Shuffle the dataset



    imgs = np.array(imgs)

    lbls = np.array(lbls)

    if isDL ==True:

        lbls = to_categorical(lbls) # for keras CNN Model

    return imgs, lbls
def display_prediction(col_size, row_size,XPred,yPred): 

    img_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(XPred[img_index][:,:,0], cmap='gray')

            ax[row][col].set_title("({}) {}".format(yPred[img_index],get_classlabel(yPred[img_index],'Ar')))

            ax[row][col].set_xticks([])

            ax[row][col].set_yticks([])

            img_index += 1
# Split our Dataset into Training, Test and Prediction

# take 3 images per class for later prediction (96 images 3 x 32 class category)

# split the remaining into 20% test and 80% Training



dtTraining, dtTest,dtPred = SplitData(3,0.3)
print('Pred     {} \t # {} per class'.format(dtPred.shape[0], dtPred.shape[0] //32))

print('Training {} \t # {} per class'.format(dtTraining.shape[0], dtTraining.shape[0] //32))

print('Test     {} \t # {} per class'.format(dtTest.shape[0], dtTest.shape[0] //32))

print('---------------')

print('Sum      {}'.format(dtTraining.shape[0] + dtTest.shape[0] + dtPred.shape[0]))
ddata = {"Training":dtTraining.groupby("ClassId").size(),"Test":dtTest.groupby("ClassId").size()}

iindex = range(32)



ddataframe = pd.DataFrame(data=ddata, index= iindex)

ddataframe.plot.bar(stacked= True, rot= 15, title='Training vs Test data')

plt.show(block= True)
X_train,y_train = getDataSet('Training',False)

X_Valid,y_valid = getDataSet('Test',False)

X_pred,_ = getDataSet('Prediction',False)
print("Shape of Train Images:{} , Train Labels: {}".format(X_train.shape,y_train.shape))
from sklearn.neighbors import KNeighborsClassifier

# knn_classifier = KNeighborsClassifier(algorithm=, n_jobs=-1)

knn_classifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

                                      metric_params=None, n_jobs=-1, n_neighbors=5, p=2,weights='uniform')

knn_classifier.fit(X_train, y_train)
knnScore = knn_classifier.score(X_train, y_train)

print(knnScore)
from sklearn.metrics import confusion_matrix

y_ValidPrediction = knn_classifier.predict(X_Valid)

# Convert predictions classes to one hot vectors 

Y_pred_classes = y_ValidPrediction



Y_true = y_valid

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



plt.figure(figsize=(10,8))

sns.heatmap(confusion_mtx, annot=True, fmt="d");
knn_predictions = knn_classifier.predict(X_pred)

print(knn_predictions)
X_train,y_train = getDataSet('Training',True)

X_valid,y_valid = getDataSet('Test',True)

X_pred,_ = getDataSet('Prediction',True)
print("Shape of Train Images:{} , Train Labels: {}".format(X_train.shape,y_train.shape))

print("Shape of Test Images:{} , Test Labels: {}".format(X_valid.shape,y_valid.shape))

print("Shape of Prediction Images:{} , Prediction Labels: {}".format(X_pred.shape,"?"))
model = Models.Sequential()



model.add(Layers.Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1)))

model.add(Layers.Conv2D(64, (3, 3), activation='relu'))

model.add(Layers.MaxPooling2D(pool_size=(2, 2)))

model.add(Layers.Dropout(0.25))

model.add(Layers.Flatten())

model.add(Layers.Dense(128, activation='relu'))

model.add(Layers.Dropout(0.5))

model.add(Layers.Dense(32, activation='softmax'))



model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])



model.summary()

Utils.plot_model(model,to_file='model.png',show_shapes=True, show_layer_names=True, dpi=80)

# model = Sequential(name='Predict')

# model.add(Layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(64,64,1), name='layer_conv2',strides = (1,1), padding='same'))

# model.add(Layers.BatchNormalization())

# model.add(Layers.MaxPooling2D((2,2),name='maxPool2'))

# model.add(Layers.Conv2D(32,(3,3),strides = (1,1),name='conv3',padding='same'))

# model.add(Layers.BatchNormalization())

# model.add(Layers.MaxPooling2D((2,2),name='maxPool3'))

# model.add(Layers.Flatten())

# model.add(Layers.Dense(32,activation = 'relu',name='fc0'))

# model.add(Layers.Dropout(0.25))

# model.add(Layers.Dense(32,activation = 'relu',name='fc1'))

# model.add(Layers.Dropout(0.25))

# model.add(Layers.Dense(32,activation = 'softmax',name='fc2'))



# model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])



# model = Sequential(name='Predict')

# model.add(Layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(64,64,1), name='layer_conv2'))

# model.add(Layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name='conv3'))

# model.add(Layers.MaxPooling2D((2,2),name='maxPool1'))

# model.add(Layers.Dropout(0.25))

# model.add(Layers.Flatten())

# model.add(Layers.Dense(128,activation = 'relu',name='fc0'))

# model.add(Layers.Dropout(0.5))

# model.add(Layers.Dense(32,activation = 'softmax',name='fc2'))



# model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
# model = Sequential(name='Predict')

# model.add(Layers.Conv2D(64, kernel_size=(3, 3), activation='relu',input_shape=(64,64,1), name='conv2'))

# model.add(Layers.MaxPooling2D((2,2),name='maxPool1'))

# model.add(Layers.Flatten())

# model.add(Dense(128, activation='relu'))

# model.add(Layers.Dropout(0.2))

# model.add(Layers.Dense(32,activation = 'softmax',name='dns1'))



# model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
callbacks_list =[EarlyStopping(monitor='val_loss', patience=10), ModelCheckpoint(

    filepath='model_255.h5', monitor='val_loss', save_best_only= True),]



trained = model.fit(X_train, y_train, epochs=35, validation_data=(X_valid, y_valid), 

                    callbacks= callbacks_list)

plt.plot(trained.history['accuracy'])

plt.plot(trained.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



plt.plot(trained.history['loss'])

plt.plot(trained.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
print("on Validation data")

pred1=model.evaluate(X_valid,y_valid)

print("accuaracy", str(pred1[1]*100))

print("Total loss",str(pred1[0]*100))
from sklearn.metrics import confusion_matrix

Y_prediction = model.predict(X_valid)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_prediction,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_valid,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 



plt.figure(figsize=(10,8))

plt.title('Validation confusion_matrix', fontsize = 16) 

sns.heatmap(confusion_mtx, annot=True, fmt="d");

cnn_Y_pred = model.predict(X_pred)

cnn_Y_pred = np.argmax(cnn_Y_pred,axis = 1)

print(cnn_Y_pred)
def display_prediction(col_size, row_size,XPred,yPred): 

    img_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(X_pred[img_index][:,:,0], cmap='gray')

            ax[row][col].set_title("({}) {}".format(yPred[img_index],get_classlabel(yPred[img_index],'Ar')))

            ax[row][col].set_xticks([])

            ax[row][col].set_yticks([])

            img_index += 1
display_prediction(12,8,X_pred,cnn_Y_pred)
# layer_outputs = [layer.output for layer in model.layers[:9]] # Extracts the outputs of the top 12 layers

# activation_model = Models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input



layer_outputs = [layer.output for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(X_train[10].reshape(1,64,64,1))

 

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index=0

    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))

    for row in range(0,row_size):

        for col in range(0,col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')

            activation_index += 1
plt.imshow(X_train[10][:,:,0],cmap='gray');
display_activation(activations, 8, 8, 1)
display_activation(activations, 8, 8, 2)
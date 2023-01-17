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
import tensorflow as tf 

bug

import matplotlib

import numpy as np



matplotlib.use("Agg")

from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Sequential

from keras.models import Model, load_model

from keras.optimizers import SGD

from sklearn.metrics import classification_report

from keras.callbacks import LearningRateScheduler

from keras.models import model_from_json



from keras import optimizers

#from configutils import config

#from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import os


from keras.preprocessing.image import ImageDataGenerator



# create a data generator

traingen = ImageDataGenerator(horizontal_flip = True, rescale = 1.0 / 255, fill_mode = "nearest", zoom_range = 0.2, shear_range = 0.2)

testgen = ImageDataGenerator(rescale = 1.0 / 255, fill_mode = "nearest", zoom_range = 0.2, shear_range = 0.2)



def layer(input) :

	j = BatchNormalization()(input)

	j = Activation('relu')(j)

	j = Conv2D(12, (3, 3), activation = 'relu', padding = 'same')(j)

	j = Dropout(0.2)(j)

	return(j)



def denseBlock(input) :

	a = layer(input)

	b = concatenate([input, a], axis = -1)

	c = layer(b)

	d = concatenate([b, c], axis = -1)

	e = layer(d)

	f = concatenate([d, e], axis = -1)

	g = layer(f)

	h = concatenate([a, c, e, g], axis = -1)

	return(h)



    

def denseNet_model(epochs = None):

    # Conv

    inputs = Input((224, 224, 3))

    i = Conv2D(48, (3, 3), activation = 'relu', padding = 'same')(inputs)



    #### DB

    j = denseBlock(i)



    # TD

    j = BatchNormalization()(j)

    j = Activation('relu')(j)

    j = Conv2D(12, (1, 1), activation = 'relu')(j)

    j = Dropout(0.2)(j)

    j = MaxPooling2D(pool_size = 2 * 2)(j)



    #### DB

    j = denseBlock(j)



    # FC

    #j = Flatten()(j)

    j = GlobalMaxPooling2D()(j)

    outputs = Dense(11, activation = 'softmax')(j) # 11 classes dans le dataset

    

    #Optimizer

    opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

    

    # Compilation du modèle

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    return model



# Dégradation par pas

def step_decay(curr_epoch):

    init_eta = 0.01

    drop_factor = 0.25

    drop_every = 10

    decay = np.floor((1 + curr_epoch) / drop_every)

    return init_eta * drop_factor ** decay





# Dégradation polynômiale

def poly_decay(curr_epoch):

    total_epochs = 100

    init_eta = 0.01

    order = 1.0

    decay = (1 - (curr_epoch / float(total_epochs))) ** order

    return init_eta * decay





#Fonction d'évaluation du modèle

def evaluate(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it = traingen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it = testgen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = denseNet_model(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score





#Evaluation du DenseNet

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluate(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluate(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]



epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluate(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]



#2) Model optimal

train_it = traingen.flow_from_directory('../input/AFF11_resized/AFF11_resized_train/', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it = testgen.flow_from_directory('../input/AFF11_resized/AFF11_resized_test/', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = denseNet_model(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_DenseNet.json", "w") as file:

    file.write(json)

model.save_weights("model_DenseNet.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_DenseNet.json",mode='w') as f:

    history_DenseNet.to_json(f)



# Save accuracy/recall/precision/Fmeasure

#filenames = test_it.filenames

#nb_samples = len(filenames)

#predict = model.predict_generator(test_it,steps = 

#                                   np.ceil(nb_samples/batch_size))

#labelNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

#print(classification_report(0,

# predict.argmax(axis=1), target_names=labelNames))



#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_DenseNet.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_DenseNet.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_DenseNet.csv", datshit, delimiter=",")



#La même chose qu'au dessus, pour Cropped...

    

#Fonction d'évaluation du modèle

def evaluate2(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it = traingen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it = testgen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = denseNet_model(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du DenseNet

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluate2(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluate2(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluate2(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]



#2) Model optimal

train_it = traingen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it = testgen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = denseNet_model(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_DenseNet2.json", "w") as file:

    file.write(json)

model.save_weights("model_DenseNet2.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_DenseNet2.json",mode='w') as f:

    history_DenseNet.to_json(f)



# Save accuracy/recall/precision/Fmeasure

#filenames = test_it.filenames

#nb_samples = len(filenames)

#predict = model.predict_generator(test_it,steps = 

#                                   np.ceil(nb_samples/batch_size))

#labelNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

#print(classification_report(0,

# predict.argmax(axis=1), target_names=labelNames))



#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_DenseNet2.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_DenseNet2.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_DenseNet2.csv", datshit, delimiter=",")

#Partie B

# pip install git+https://github.com/qubvel/classification_models.git



from classification_models.keras import *



#1 Les images en entrée vont être normalisée par rapport à la moyenne des plans RGB des images de imageNet

#2 Appliquer une augmentation des données de train, avec un flip horizontal



normalizegen = ImageDataGenerator(rescale=1.0/255.0)

normalize2gen = ImageDataGenerator(rescale=1.0/255.0,horizontal_flip = True)



mean = np.array([123.68, 116.779, 103.939], dtype="float32")

normalizegen.mean = mean

normalize2gen.mean = mean





    
def resnet18_model_1FT(epochs = None):

        #3 Charger un modèle pré-appris sur ImageNet sans la dernière couche FC

        ResNet18, preprocess_input = Classifiers.get('resnet18')

        myResNet18_model=ResNet18(input_shape=(224,224,3),weights='imagenet', classes=1000, include_top=False)



        #5 Définir une nouvelle couche FC identique à l’ancienne 

        top_model = myResNet18_model.output

        top_model=GlobalAveragePooling2D(data_format=None, name='pool1')(top_model)

        top_model=Dense(11,kernel_initializer='random_uniform',name='fc1')(top_model)

        top_model=Activation('softmax',name='softmax')(top_model)



        #6 Reconstruire le nouveau modèle. 

        my_model = Model(inputs=myResNet18_model.input, outputs=top_model)



        for layer in my_model.layers[:-3]:

            layer.trainable = False

            

        #Optimizer

        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

    

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model



def resnet18_model_2FT(epochs = None):

        #a) load model optimal

        json_file = open('optimal_resnet18_1FT.json', 'r')

        loaded_model_json = json_file.read()

        json_file.close()

        my_model = model_from_json(loaded_model_json)

        # charger les poids dans dans un nouveau modele

        my_model.load_weights("model_resnet18_1FT.h5") 

        #a) Geler la première couche et faire un fine-tuning sur les couches supérieures. 



        for layer in my_model.layers:

            layer.trainable = True

    

        for layer in my_model.layers[:+18]:

            layer.trainable = False

    

        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model



def resnet18_model_3FT(epochs = None):

        #a) load model optimal

        json_file = open('optimal_resnet18_1FT.json', 'r')

        loaded_model_json = json_file.read()

        json_file.close()

        my_model = model_from_json(loaded_model_json)

        # charger les poids dans dans un nouveau modele

        my_model.load_weights("model_resnet18_1FT.h5") 

        #b) Geler les deux premières couches et faire un fine-tuning sur les couches supérieures

        for layer in my_model.layers:

            layer.trainable = True



        for layer in my_model.layers[:+30]:

            layer.trainable = False



        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model
#Fonction d'évaluation du modèle

def evaluateResnet18_1FT(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_1FT(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_1FT(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]





batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_1FT(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_1FT(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_1FT(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_1FT.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_1FT.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_1FT.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_1FT.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_1FT.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_1FT.csv", datshit, delimiter=",")

#Fonction d'évaluation du modèle

def evaluateResnet18_2FT(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_2FT(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_2FT(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradation[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_2FT(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_2FT(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_2FT(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_2FT.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_2FT.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_2FT.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_2FT.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_2FT.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_2FT.csv", datshit, delimiter=",")



#Fonction d'évaluation du modèle

def evaluateResnet18_3FT(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_3FT(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_3FT(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradation[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_3FT(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_3FT(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/resized/AFF11_resized/AFF11_resized_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_3FT(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_3FT.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_3FT.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_3FT.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_3FT.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_3FT.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_3FT.csv", datshit, delimiter=",")



def resnet18_model_1FT2(epochs = None):

        #3 Charger un modèle pré-appris sur ImageNet sans la dernière couche FC

        ResNet18, preprocess_input = Classifiers.get('resnet18')

        myResNet18_model=ResNet18(input_shape=(224,224,3),weights='imagenet', classes=1000, include_top=False)



        #5 Définir une nouvelle couche FC identique à l’ancienne 

        top_model = myResNet18_model.output

        top_model=GlobalAveragePooling2D(data_format=None, name='pool1')(top_model)

        top_model=Dense(11,kernel_initializer='random_uniform',name='fc1')(top_model)

        top_model=Activation('softmax',name='softmax')(top_model)



        #6 Reconstruire le nouveau modèle. 

        my_model = Model(inputs=myResNet18_model.input, outputs=top_model)



        for layer in my_model.layers[:-3]:

            layer.trainable = False

            

        #Optimizer

        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

    

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model



def resnet18_model_2FT2(epochs = None):

        #a) load model optimal

        json_file = open('optimal_resnet18_1FT2.json', 'r')

        loaded_model_json = json_file.read()

        json_file.close()

        my_model = model_from_json(loaded_model_json)

        # charger les poids dans dans un nouveau modele

        my_model.load_weights("model_resnet18_1FT2.h5") 

        #a) Geler la première couche et faire un fine-tuning sur les couches supérieures. 



        for layer in my_model.layers:

            layer.trainable = True

    

        for layer in my_model.layers[:+18]:

            layer.trainable = False

    

        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model



def resnet18_model_3FT2(epochs = None):

        #a) load model optimal

        json_file = open('optimal_resnet18_1FT2.json', 'r')

        loaded_model_json = json_file.read()

        json_file.close()

        my_model = model_from_json(loaded_model_json)

        # charger les poids dans dans un nouveau modele

        my_model.load_weights("model_resnet18_1FT2.h5") 

        #b) Geler les deux premières couches et faire un fine-tuning sur les couches supérieures

        for layer in my_model.layers:

            layer.trainable = True



        for layer in my_model.layers[:+30]:

            layer.trainable = False



        opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs if epochs else 0.0, nesterov=False)

        my_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])



        return my_model

    
#Fonction d'évaluation du modèle

def evaluateResnet18_1FT2(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_1FT2(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_1FT2(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_1FT2(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_1FT2(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_1FT2(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_1FT2.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_1FT2.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_1FT2.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_1FT2.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_1FT2.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_1FT2.csv", datshit, delimiter=",")

#Fonction d'évaluation du modèle

def evaluateResnet18_2FT2(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_2FT2(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_2FT2(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]



batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_2FT2(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_2FT2(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_2FT2(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_2FT2.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_2FT2.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_2FT2.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_2FT2.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_2FT2.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_2FT2.csv", datshit, delimiter=",")
#Fonction d'évaluation du modèle

def evaluateResnet18_3FT2(epochs, batch_size=32, degradation=None, checkpoint=False):

    train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

    test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



    array = []

    if checkpoint:

        filepath="/kaggle/working/best-{epoch:02d}-{val_accuracy:.2f}.hdf5"

        array = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        array.append(checkpoint)

    if degradation == "pas":

        array.append(LearningRateScheduler(step_decay))

    if degradation == "poly":

        array.append(LearningRateScheduler(poly_decay))

        

    model = resnet18_model_3FT2(epochs=epochs if degradation == "standard" else None)

    history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)

    score = model.evaluate_generator(generator=test_it, steps=test_it.n//test_it.batch_size)

    

    return model, history, score



#Evaluation du resnet18 avec nouvelle FC

degradations = ["standard", "pas", "poly"]

degradation_loss = [evaluateResnet18_3FT2(30, 32, degradation)[2][0] for degradation in degradations]

degradation = degradations[np.argmin(degradation_loss)]



degradation="standard"

batch_size = [8, 16, 32]

batch_size_loss = [evaluateResnet18_3FT2(30, batch, degradation)[2][0] for batch in batch_size]

batch_size = batch_size[np.argmin(batch_size_loss)]





epochs = [10, 25,50,75,100,150]

epochs_loss = [evaluateResnet18_3FT2(epoch, batch_size, degradation)[2][0] for epoch in epochs]

epochs = epochs[np.argmin(epochs_loss)]





#2) Model optimal

train_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_train', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it =  normalizegen.flow_from_directory('../input/cropped/AFF11_crops/AFF11_crops_test', class_mode = 'categorical', batch_size = batch_size, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



array = []

if degradation == "pas":

    array.append(LearningRateScheduler(step_decay))

if degradation == "poly":

    array.append(LearningRateScheduler(poly_decay))

model = resnet18_model_3FT2(epochs=epochs if degradation == "standard" else None)

history = model.fit_generator(train_it, steps_per_epoch = train_it.n//batch_size, validation_data = test_it, validation_steps = test_it.n//batch_size, epochs=epochs, callbacks=array)



# sauvegarde du model

json = model.to_json()

with open("optimal_resnet18_3FT2.json", "w") as file:

    file.write(json)

model.save_weights("model_resnet18_3FT2.h5") 

    

# sauvegarde de l'historique d'apprentissage du model

history_DenseNet = pd.DataFrame(history.history) 

with open("history_resnet18_3FT2.json",mode='w') as f:

    history_DenseNet.to_json(f)





#Graphes 

N = np.arange(0, epochs)

plt.style.use("ggplot")

plt.figure()

plt.plot(N, history.history["loss"], label="train_loss")

plt.plot(N, history.history["val_loss"], label="val_loss")

plt.plot(N, history.history["accuracy"], label="train_acc")

plt.plot(N, history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy (Simple NN)")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.savefig("Loss_Acc_mnist") 



#3 Utiliser le modèle comme extracteur de caract

#a) load model optimal

json_file = open('optimal_resnet18_3FT2.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# charger les poids dans dans un nouveau modele

loaded_model.load_weights("model_resnet18_3FT2.h5") 



# on met les labels (variables à prédire) dans 'labels'

labels = train_it.labels



# on extrait tout ce qu'il y a dans l'avant dernière couche du model "model"

extract = Model(loaded_model.inputs, loaded_model.layers[len(loaded_model.layers) - 2].output)

features = extract.predict(train_it)



datshit = []

# 589 == le nombre d'images dans "train_it"

for i in range (0, 589) :

    # ici très savant mélange pour combiner les vecteurs descripteurs et les labels dans un dataframe

    arr1 = np.array([features[i]])

    arr2 = np.array([labels[i]])

    arr_flat = np.append(arr1, arr2)

    datshit.append(arr_flat)



np.savetxt("descripteurs_resnet18_3FT2.csv", datshit, delimiter=",")
import numpy as np 

import pandas as pd



from sklearn import svm

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



class Dataset : # kernels : "rbf", "poly" ou "sigmoid" # gamma : "scale" ou "auto" # algorith : "ball_tree", "kd_tree" ou "brute"

    def __init__(self, filename, kernel = "linear", gamma = "auto", n_neighbors = 5, algorithm = "auto") :

        data = pd.read_csv(filename, header = 0)

        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0 : len(data.columns) - 2], data.iloc[:, len(data.columns) - 1], test_size = 0.5)

        my_SVM = svm.SVC(kernel = kernel, gamma = gamma).fit(x_train, y_train)

        my_KNN = KNeighborsClassifier(n_neighbors = n_neighbors, algorithm = algorithm).fit(x_train, y_train)

        self.filename = filename

        self.accuSVM, self.confusionSVM, self.reportSVM = self.results(y_test, my_SVM.predict(x_test), filename)

        self.accuKNN, self.confusionKNN, self.reportKNN = self.results(y_test, my_KNN.predict(x_test), filename)



    def results(self, y_test, y_pred, filename) :

        return round(accuracy_score(y_test, y_pred) * 100, 2), confusion_matrix(y_test, y_pred, labels = np.unique(y_pred)), classification_report(y_test, y_pred, labels = np.unique(y_pred))



    def showEverything(self) :

        print("\n\n----------", self.filename, "----------")

        print("\n\n----- SVM -----\n\nConfusion Matrix :\n", self.confusionSVM, "\n\nClassification Report :\n", self.reportSVM, "\n\nAccuracy :\n", self.accuSVM, "%")

        print("\n\n----- KNN -----\n\nConfusion Matrix :\n", self.confusionKNN, "\n\nClassification Report :\n", self.reportKNN, "\n\nAccuracy :\n", self.accuKNN, "%")

        

    def acc(self) :

        print("\n\n----------", self.filename, "----------")

        print("\nSVM :", self.accuSVM, "%   |   KNN :", self.accuKNN, "%")



#data = Dataset("descripteurs_resnet18_1FT.csv")

#data.acc()

#data = Dataset("descripteurs_resnet18_3FT.csv")

#data.acc()

#data = Dataset("descripteurs_resnet18_3FT2.csv")

#data.acc()

#print(data.accuSVM)
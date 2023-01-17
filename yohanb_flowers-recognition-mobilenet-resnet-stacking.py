# LA

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, StratifiedKFold



# Keras

from keras.preprocessing import image

from keras import applications

from keras.models import Model

from keras import layers, initializers, optimizers, regularizers

from keras import datasets, models, callbacks, applications, utils



#  Supress warnings

import warnings

warnings.filterwarnings("always")

warnings.filterwarnings('ignore')



# Misc.

import os

import glob

img_path_train = '../input/train/train'

img_path_test = '../input/test/test'
# Load the labels:

labels = pd.read_csv('../input/{}'.format('train_labels.csv'))



# Encoding:

le = LabelEncoder(); 

labels['Classes'] = le.fit_transform(labels['Category'])



# Show classes:

for i, c in enumerate(le.classes_): 

    print(i, '=', c)

    

labels.head(3)
# Load the data for Mobilenet:

# ----------------------------------

# Load train data:

X = []

for image_path in np.sort(glob.glob(img_path_train + "/*.jpg")):

    img = image.load_img(image_path, target_size=(224, 224))

    img_data = image.img_to_array(img)

    img_data = applications.mobilenet.preprocess_input(img_data)

    X.append(img_data)



X = np.array(X)

y = labels.Classes.values

# ----------------------------------

# Load test data:

X_test = []

for image_path in np.sort(glob.glob(img_path_test + "/*.jpg")):

    img = image.load_img(image_path, target_size=(224, 224))

    img_data = image.img_to_array(img)

    img_data = applications.mobilenet.preprocess_input(img_data)

    X_test.append(img_data)



X_test = np.array(X_test)

# ----------------------------------



print('Data has been loaded.\n')



# Generating the folds:

cv_n = 5 # number of folders

folds = StratifiedKFold(n_splits=cv_n, random_state=42).split(X, y)

pred_1 = np.zeros((X.shape[0], 5)) # empty array for Stage 1 predictions



# Getting Stage 1 predictions on Train:

for train_index, test_index in folds:

    

    # Model 1

    # ---------------------------

    # pretrained model without top layer

    base_model1 = applications.MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    #fix weights

    base_model1.trainable = False



    # and our model become simple

    inp = layers.Input((224, 224, 3))

    mobilenet = base_model1(inp)



    gap = layers.GlobalAveragePooling2D()(mobilenet)

    fc = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.1))(gap)

    #fc = layers.Dense(64, activation='relu')(fc)

    do = layers.Dropout(0.4)(fc)

    fc = layers.Dense(5, activation='softmax')(do)



    model1 = models.Model(inp, fc)



    model1.compile(optimizer='adam', 

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])



    # train only dense layer on top

    model1.fit(X[train_index], y[train_index],

              batch_size=16,

              epochs=1,

              verbose=1, 

              callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])

    

    # unfreeze all weights and train 

    base_model1.trainable = True

    model1.compile(optimizer=optimizers.Adam(1e-4), 

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])



    model1.fit(X[train_index], y[train_index],

              batch_size=16,

              epochs=10,

              verbose=1, 

              callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])

    # ---------------------------

    

    pred_part = model1.predict(X[test_index])

    pred_1[test_index] = pred_part

    print('\nOOB accurasy =', accuracy_score(pred_part.argmax(axis=1), y[test_index]), end='\n\n')



print('\nTolal accurasy on train=', accuracy_score(pred_1.argmax(axis=1), y), end='\n\n')



# Getting Stage 1 predictions on Test:

base_model1.trainable = False



# Model 1

# ---------------------------

# train only dense layer on top

model1.fit(X, y,

           batch_size=16,

           epochs=1,

           verbose=1, 

           callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])



# unfreeze all weights and train 

base_model1.trainable = True

model1.fit(X, y,

           batch_size=16,

           epochs=10,

           verbose=1, 

           callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])

# ---------------------------



pred_test_1 = model1.predict(X_test)



print('\nStage 1 predictions on Test has been obtained.')
# Load the data for ResNet50:

# ----------------------------------

# Load train data:

X = []

ids = []

for image_path in np.sort(glob.glob(img_path_train + "/*.jpg")):

    img = image.load_img(image_path, target_size=(224, 224))

    img_data = image.img_to_array(img)

    img_data = applications.resnet50.preprocess_input(img_data)

    X.append(img_data)

    ids.append(image_path.split('/')[-1][:-4]) # load ids

    

X = np.array(X)

ids = np.array(ids) # save ids

y = labels.Classes.values

# ----------------------------------

# Load test data:

X_test = []

ids_test = []

for image_path in np.sort(glob.glob(img_path_test + "/*.jpg")):

    img = image.load_img(image_path, target_size=(224, 224))

    img_data = image.img_to_array(img)

    img_data = applications.resnet50.preprocess_input(img_data)

    X_test.append(img_data)

    ids_test.append(image_path.split('/')[-1][:-4]) # load ids

    

X_test = np.array(X_test)

ids_test = np.array(ids_test) # save ids

# ----------------------------------



print('Data has been loaded.\n')



# Generating the folds:

cv_n = 5 # number of folders

folds = StratifiedKFold(n_splits=cv_n, random_state=42).split(X, y)

pred_2 = np.zeros((X.shape[0], 5)) # empty array for Stage 1 predictions



# Getting Stage 1 predictions on Train:

for train_index, test_index in folds:

    

    # Model 2

    # ---------------------------

    #pretrained model without top layer

    base_model2 = applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

    #fix weights

    base_model2.trainable = False



    #and our model become simple

    inp = layers.Input((224, 224, 3))

    resnet = base_model2(inp)



    fc = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.2))(resnet)

    do = layers.Dropout(0.3)(fc)

    fc = layers.Dense(5, activation='softmax')(do)



    model2 = models.Model(inp, fc)



    model2.compile(optimizer='adam', 

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])



    #train only dense layer on top

    model2.fit(X[train_index], y[train_index],

              batch_size=16,

              epochs=1,

              verbose=1, 

              callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])



    #unfreeze all weights and train 

    base_model2.trainable = True

    model2.compile(optimizer=optimizers.Adam(1e-4), 

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])



    model2.fit(X[train_index], y[train_index],

              batch_size=16,

              epochs=10,

              verbose=1, 

              callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])

    # ---------------------------

    

    pred_part = model2.predict(X[test_index])

    pred_2[test_index] = pred_part

    print('\nOOB accurasy =', accuracy_score(pred_part.argmax(axis=1), y[test_index]), end='\n\n')



print('\nTolal accurasy on train=', accuracy_score(pred_2.argmax(axis=1), y), end='\n\n')



# Getting Stage 1 predictions on Test:

base_model2.trainable = False



# Model 2

# ---------------------------

# train only dense layer on top

model2.fit(X, y,

           batch_size=16,

           epochs=1,

           verbose=1, 

           callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])



# unfreeze all weights and train 

base_model2.trainable = True

model2.fit(X, y,

           batch_size=16,

           epochs=10,

           verbose=1, 

           callbacks=[callbacks.ReduceLROnPlateau(patience=2, verbose=1)])

# ---------------------------



pred_test_2 = model2.predict(X_test)



print('\nStage 1 predictions on Test has been obtained.')
pred = np.hstack((pred_1, pred_2, pred_1*pred_2, pred_1/pred_2))

X_st2 = pd.DataFrame(pred, columns=['f0','f1','f2','f3','f4','q0','q1','q2','q3','q4','fq0','fq1','fq2','fq3','fq4', 'f:g0', 'f:g1', 'f:g2', 'f:g3', 'f:g4'])

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

print(cross_val_score(rf, X_st2, y, cv=5, scoring='accuracy')) 

rf.fit(X_st2, y)
pred_test = np.hstack((pred_test_1, pred_test_2, pred_test_1*pred_test_2, pred_test_1/pred_test_2))

X_test_st2 = pd.DataFrame(pred_test, columns=['f0','f1','f2','f3','f4','q0','q1','q2','q3','q4','fq0','fq1','fq2','fq3','fq4', 'f:g0', 'f:g1', 'f:g2', 'f:g3', 'f:g4'])

pred_sub_test = pd.DataFrame({'Id':ids_test, 'Category': rf.predict(X_test_st2)})

pred_sub_test['Category'] = pred_sub_test.Category.map({0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'})



# Save output

pred_sub_test.to_csv('csv_test.csv', index=False)
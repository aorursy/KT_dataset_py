import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import keras
import pandas as pd
import pickle
from keras import regularizers
DIR = '../input'
training_dir = os.path.join(DIR, 'train')
test_path = os.path.join(DIR, 'test')
categories = [training_dir, test_path]
def CreateXY(categories, img_size):
#categories: a list with training and testing directories; training first; example: categories[0] -> 'C:/Users/Leo/Desktop/all\\train'
    X_train = []
    y_train = []
    test_set = []
    
    train_cat_dir = []
    train_dog_dir = []
    test_dir = []

    #training
    for img in os.listdir(categories[0]):
        if img.startswith('cat'):
            train_cat_dir.append(os.path.join(categories[0], img))
        else:
            train_dog_dir.append(os.path.join(categories[0], img))
            
    for img in train_cat_dir:
        cat = cv2.imread(img, 0)
        cat = cv2.resize(cat, (img_size,img_size))
        cat = np.divide(cat,255)
        X_train.append(cat)
        y_train.append(0)
        
    for img in train_dog_dir:
        dog = cv2.imread(img, 0)
        dog = cv2.resize(dog, (img_size,img_size))
        dog = np.divide(dog,255)
        X_train.append(dog)
        y_train.append(1)
        
    for img in os.listdir(categories[1]):
        test_dir.append(os.path.join(categories[1], img))
        
    for img in test_dir:
        und = cv2.imread(img, 0)
        und = cv2.resize(und, (img_size,img_size))
        und = np.divide(und,255)
        test_set.append(und)
        
    return X_train, y_train, test_set
X_train, y_train, test_set = CreateXY(categories, 50)
len(X_train)
len(y_train)
X_train_shuff = []
y_train_shuff = []
index_shuf = [val for val in range(len(X_train))]
np.random.shuffle(index_shuf)
for i in index_shuf:
    X_train_shuff.append(X_train[i])
    y_train_shuff.append(y_train[i])
X_train = np.array(X_train_shuff)
y_train = np.array(y_train_shuff)
test_set = np.array(test_set)
#I'm doing it 'cause I saw in an answer that it should help
X_train = X_train.reshape(-1,50,50,1)
test_set = test_set.reshape(-1,50,50,1)
#I've created my own architecture, maybe that's the reason of the acc. problem
model = keras.models.Sequential()
#Input shape problems
model.add(keras.layers.Conv2D(32, kernel_size=(5,5), input_shape=X_train.shape[1:]))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Conv2D(32, kernel_size=(5,5)))
model.add(keras.layers.MaxPooling2D(pool_size=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
#added dropout and l2 regularization
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150)
preds = model.predict_classes(test_set)
preds = [val[0] for val in preds]
test_names = [int(val.split('.')[0]) for val in os.listdir(categories[1])]
id_series = pd.Series(data=test_names, name='id')
label_series = pd.Series(data=preds, name='label')
df = pd.DataFrame({'id':id_series, 'label':label_series})
df.to_csv('submission.csv', index=False)


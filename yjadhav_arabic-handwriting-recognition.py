import os
import cv2
import pandas as pd
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PIXELS = 1024
DIMENSIONS = np.int16(math.sqrt(PIXELS))

TRAINING_FEATURES_FILE = "csvTrainImages 13440x1024.csv"
TRAINING_LABELS_FILE = "csvTrainLabel 13440x1.csv"
TESTING_FEATURES_FILE = "csvTestImages 3360x1024.csv"
TESTING_LABELS_FILE = "csvTestLabel 3360x1.csv"

def load_data(file=TRAINING_FEATURES_FILE, header=True):
    csv_path = os.path.join("dataset/", file)
    if header:
        return pd.read_csv(csv_path)
    else:
        return pd.read_csv(csv_path, header=None)
data = load_data(TRAINING_FEATURES_FILE)
data.head()
from matplotlib import pyplot as plt

def imagify(arr, getimage=False, showimage=True):
    img = np.array(np.reshape(arr, (DIMENSIONS, DIMENSIONS)), dtype="uint8")
    if showimage:
        plt.imshow(img, interpolation='nearest')
        plt.gray()
        plt.show() 
        
    if getimage:
        return img
def showimage(img):
    plt.imshow(img, interpolation='nearest')
    plt.gray()
    plt.show() 
img = imagify(data.values[7], getimage=True)
th1,img1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
print(th1)
showimage(img1)
th2,img2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(th2)
showimage(img2)
blur = cv2.GaussianBlur(img,(5,5),0)
th3,img3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(th3)
showimage(img3)
THRESH_BINARY = cv2.THRESH_BINARY
THRESH_BINARY_AND_THRESH_OTSU = cv2.THRESH_BINARY+cv2.THRESH_OTSU
def apply_thresholding(df, cap=0, thres=THRESH_BINARY_AND_THRESH_OTSU):
    if thres == None:
        return df
    
    values = df.values
    thres_values = []
    thresholding_started = False
    for value in values:
        img = imagify(value, getimage=True, showimage=False)
        th_,img = cv2.threshold(img,cap,255,thres)
        img = [img.flatten()]
        if thresholding_started:
            thres_values = np.concatenate((thres_values, img), axis=0)
        else:
            thres_values = img
            thresholding_started = True
            
    thres_df = pd.DataFrame(thres_values, columns=df.columns)
    return thres_df
datacopy = data.copy()
data = apply_thresholding(data, thres=THRESH_BINARY_AND_THRESH_OTSU)
data.head()
from sklearn.decomposition import PCA
import numpy as np

def get_dims_variances(x, min_dim, max_dim, threshold=0.1, capToThreshold=False):
    dims = []
    variances = []
    optimum_dim = min_dim
    saturation_reached = False
    for dim in range(min_dim, max_dim + 1):
        pca = PCA(n_components=dim)
        pca.fit(x)
        variance = np.array(pca.explained_variance_ratio_)
        variance = variance.min()
        if threshold < variance:
            optimum_dim = dim
        else:
            saturation_reached = True
        
        if saturation_reached and capToThreshold:
            break
        else:    
            dims.append(dim)
            variances.append(variance)
        
    return dims, variances, optimum_dim
#dims, variances, OPTIMUM_DIMENSION = get_dims_variances(data, 2, 100, 0.005, capToThreshold=True)
OPTIMUM_DIMENSION = 36
print(OPTIMUM_DIMENSION)
import matplotlib.pyplot as plt
#plt.plot(dims, variances)
#plt.show()
#dim_var = pd.DataFrame()
#dim_var["DIM"] = dims
#dim_var["VAR"] = variances
pca = PCA(n_components=OPTIMUM_DIMENSION)
training_features = pca.fit_transform(data)
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
training_features = imputer.fit_transform(training_features)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
training_features = scalar.fit_transform(training_features)
data_labels = load_data(TRAINING_LABELS_FILE)
training_labels = data_labels.values.flatten()
test_data = load_data(TESTING_FEATURES_FILE)
test_data = apply_thresholding(test_data, thres=THRESH_BINARY_AND_THRESH_OTSU)
testing_features = pca.transform(test_data)
testing_features = imputer.transform(testing_features)
testing_features = scalar.transform(testing_features)
test_data_labels = load_data(TESTING_LABELS_FILE)
testing_labels = test_data_labels.values.flatten()
# SGD Classifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

sgd_clf = SGDClassifier(random_state=42)
print("Cross Val Scores on training set\n", cross_val_score(clone(sgd_clf), training_features, training_labels, cv=3, scoring="accuracy"))


sgd_clf.fit(training_features, training_labels)
print("\n\nAccuracy on testing data set\n", sum(testing_labels == sgd_clf.predict(testing_features)) / len(testing_labels))
# KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier 

knn_clf = KNeighborsClassifier()
print("Cross Val Scores on training set\n", cross_val_score(clone(knn_clf), training_features, training_labels, cv=3, scoring="accuracy"))

knn_clf.fit(training_features, training_labels)
print("\n\nAccuracy on testing data set\n", sum(testing_labels == knn_clf.predict(testing_features)) / len(testing_labels))
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier 

forest_clf = RandomForestClassifier(random_state=42)
print("Cross Val Scores on training set\n", cross_val_score(clone(forest_clf), training_features, training_labels, cv=3, scoring="accuracy"))

forest_clf.fit(training_features, training_labels)
print("\n\nAccuracy on testing data set\n", sum(testing_labels == forest_clf.predict(testing_features)) / len(testing_labels))
# MLP Classifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer

batch_size = 128
num_classes = 28
epochs = 30

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(OPTIMUM_DIMENSION,)))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

binarizer = LabelBinarizer()
binarizer.fit(training_labels)
training_labels = binarizer.transform(training_labels)
testing_labels = binarizer.transform(testing_labels)

history = model.fit(training_features, training_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(testing_features, testing_labels))

score = model.evaluate(testing_features, testing_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# CNN Classifier
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer

batch_size = 128
num_classes = 28
epochs = 30
size = np.int16(math.sqrt(OPTIMUM_DIMENSION))

model = Sequential()
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=(size, size, 1)))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

adam = Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


training_features = np.reshape(training_features, (-1, size, size, 1))
testing_features = np.reshape(testing_features, (-1, size, size, 1))

binarizer = LabelBinarizer()
binarizer.fit(training_labels)
training_labels = binarizer.transform(training_labels)
testing_labels = binarizer.transform(testing_labels)

history = model.fit(training_features, training_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(testing_features, testing_labels))

score = model.evaluate(testing_features, testing_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

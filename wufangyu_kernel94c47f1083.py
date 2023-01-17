import numpy as np # matrix operations

import pandas as pd # data processing, CSV file processing



# Imports Keras for Deep Learning Model

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten,MaxPooling2D, LeakyReLU

from tensorflow.keras.models import Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# import image processor and data visualization tool

import cv2

from matplotlib import pyplot as plt

from glob import glob

import random



# trainning data files are in the "../input/asl_alphabet_train/asl_alphabet_train/" directory.

# test data files are in the "../input/asl_alphabet_train/asl_alphabet_train/" directory.
def plot_samples(letter):

    print("check images for letter " + letter)

    base_path = '../input/asl_alphabet_train/asl_alphabet_train/'

    img_path = base_path + letter + '/**'

    all_contents = glob(img_path)

    

    plt.figure(figsize = (16,16))

    imgs = random.sample(all_contents, 3)

    

    print("image shape: " + str(cv2.imread(imgs[0]).shape))

    

    plt.subplot(1,3,1)

    plt.imshow(cv2.imread(imgs[0]))

    plt.subplot(1,3,2)

    plt.imshow(cv2.imread(imgs[1]))

    plt.subplot(1,3,3)

    plt.imshow(cv2.imread(imgs[2]))

    return



plot_samples('A')

    
data_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train/"

target_size = (64, 64)

target_dims = (64, 64, 3)

num_classes = 29



data_augmentor = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, validation_split=0.3)

train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=64, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=64, subset="validation")
def create_model():

    my_model = Sequential()

    my_model.add(Conv2D(32, kernel_size=2, strides=1, input_shape=target_dims, padding="SAME"))

    my_model.add(LeakyReLU())

    my_model.add(MaxPooling2D(pool_size=(2,2)))

    my_model.add(Dropout(0.5))

    my_model.add(Conv2D(128, kernel_size=3, strides=1, padding="SAME"))

    my_model.add(LeakyReLU())

    my_model.add(MaxPooling2D(pool_size=(3,3)))

    my_model.add(Dropout(0.5))

    my_model.add(Conv2D(128, kernel_size=4, strides=1, padding="SAME"))

    my_model.add(LeakyReLU())

    my_model.add(MaxPooling2D(pool_size=(4,4)))

    my_model.add(Dropout(0.5))

    my_model.add(Conv2D(256, kernel_size=4, strides=1, padding="SAME"))

    my_model.add(LeakyReLU())

    my_model.add(MaxPooling2D(pool_size=(2,2)))

    my_model.add(Flatten())

    my_model.add(Dropout(0.5))

    my_model.add(Dense(512, activation='relu'))

    my_model.add(Dense(num_classes, activation='softmax'))

    my_model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=["accuracy"])

    return my_model
cur_model = create_model()
cur_model.fit_generator(train_generator, epochs=30, validation_data=val_generator)
cur_model.save('cnn_model')
img = cv2.imread('../input/test-img-sets/test4.jpeg')

plt.imshow(img)

plot_samples('A')
test_img = cv2.resize(img, (64,64), interpolation = cv2.INTER_CUBIC)

plt.imshow(test_img)

test_img = np.array([test_img])

test_img.shape
test_label = cur_model.predict_classes(test_img)



# check the probability of predicted class

print(cur_model.predict(test_img)[0][[cur_model.predict_classes(test_img)[0]]])



for character, label in train_generator.class_indices.items():

    if label == test_label:

        print('cnn_model result: ' + character)
# import SVM packages from sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn import svm, metrics, datasets

from sklearn.utils import Bunch

from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread

from skimage.transform import resize

from matplotlib import pyplot

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.externals import joblib



import os

import h5py
# featurizer1: Image Moments

def f1_image_moments(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature = cv2.HuMoments(cv2.moments(image)).flatten()

    return feature



# featurizer2: Color Histogram

def f2_color_histogram(image, mask=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    cv2.normalize(hist, hist)

    return hist.flatten()



# featurizer3: Histogram of Oriented Gradients (be careful: this featurizer will generate a very large size data)

def f3_hog(image):

    hog = cv2.HOGDescriptor()

    h = hog.compute(image)

    return h.flatten()
# get the training dataset

train_path = '../input/asl_alphabet_train/asl_alphabet_train/'

train_labels = os.listdir(train_path)



# get the test dataset

test_path = '../input/asl_alphabet_test/asl_alphabet_test/'



# sort the training/test labels

train_labels.sort()

test_labels = train_labels

print('train labels: ' + str(train_labels))

print('test labels: ' + str(test_labels))



# empty lists to hold feature vectors and labels

global_features = []

labels = []



# take top 300 training data for now: (TO-DO: find a way to fit all trainning data in memeory)

train_size = 300



# loop over the training dataset

for training_name in train_labels:

    

    # get the current training label

    dir = os.path.join(train_path, training_name)

    current_label = training_name

    number_of_images = len([name for name in dir])

    

    k = 1

    

    for file in glob(dir + '/*.jpg'):



        image = cv2.imread(file)

        

        # Global Features extraction

        f1v_image_moments = f1_image_moments(image)

        f2v_color_histogram  = f2_color_histogram(image)

        

        # comment out the HOG featurizer for smaller matrix size (TO-DO: find a way to reduce HOG feature size)

        #f3v_hog = f3_hog(image)



        # Concatenate feature values

        #global_feature = np.hstack([f1v_image_moments, f2v_color_histogram,  f3v_hog])

        global_feature = np.hstack([f1v_image_moments, f2v_color_histogram])



        # update the list of labels and feature vectors

        labels.append(current_label)

        global_features.append(global_feature)

        

        k+=1

        

        if k >= train_size:

            break

            



    print("processed trainning folder:" + current_label)
test_global_features = []

test_labels = []



for file in glob(test_path + '/*.jpg'):

    image = cv2.imread(file)

    

    current_label = file[str(file).rfind('/') + 1:str(file).rfind('_test')]



    # Global Features extraction

    f1v_image_moments = f1_image_moments(image)

    f2v_color_histogram  = f2_color_histogram(image)

    f3v_hog = f3_hog(image)

    

    # Concatenate feature values

    global_feature = np.hstack([f1v_image_moments, f2v_color_histogram,  f3v_hog])

    

    # update the list of labels and feature vectors

    test_labels.append(current_label)

    test_global_features.append(global_feature)

    print("processed test folder:" + current_label)
# get feature vector size

print("feature vector size:" + str(np.array(global_features).shape))



# get training label size

print(" training Labels: " +  str(np.array(labels).shape))



# encode the target labels

targetNames = np.unique(labels)

le = LabelEncoder()

target = le.fit_transform(labels)



# normalize the feature vector

scaler = MinMaxScaler(feature_range=(0, 1))

rescaled_features = scaler.fit_transform(global_features)



# save the feature vector using hdf5

dirName = '../output'

try:

    # Create target Directory

    os.mkdir(dirName)

    print("Directory " , dirName ,  " Created ") 

except FileExistsError:

    print("Directory " , dirName ,  " already exists")



    

svm_data_exists = os.path.isfile('../output/svm_data.h5')

if svm_data_exists:

    os.remove('../output/svm_data.h5')



svm_labels_exists = os.path.isfile('../output/svm_labels.h5')

if svm_labels_exists:

    os.remove('../output/svm_labels.h5')

    

create_h5f_data = open('../output/svm_data.h5', "x")

h5f_data = h5py.File('../output/svm_data.h5', 'w')

h5f_data.create_dataset('svm_data_1', data=np.array(rescaled_features))



create_h5f_label = open("../output/svm_labels.h5", "x")

h5f_label = h5py.File('../output/svm_labels.h5', 'w')

h5f_label.create_dataset('svm_data_1', data=np.array(target))



h5f_data.close()

h5f_label.close()
# create all the machine learning models

models = []

models.append(('LR', LogisticRegression(random_state=9)))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier(random_state=9)))

models.append(('RF', RandomForestClassifier(n_estimators=100, random_state=9)))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC(random_state=9)))



train_data = np.array(global_features)

test_data = np.array(test_global_features)

train_labels = np.array(labels)

test_labels = np.array(test_labels)



# check the shape of train/test dataset

print("Train data: " + str(train_data.shape))

print("Test data: " + str(test_data.shape))

print("Train labels: " + str(train_labels.shape))

print("Test labels: " + str(test_labels.shape))
# filter all the warnings

import warnings

warnings.filterwarnings('ignore')



# variables to hold the results and names

results = []

names = []



# K-fold cross validation take k = 8

for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, train_data, train_labels, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)



# boxplot algorithm comparison

fig = pyplot.figure()

fig.suptitle('Model comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()
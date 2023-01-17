# Importing the required libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



import os

import time

import warnings

warnings.filterwarnings("ignore")



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import tensorflow as tf

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
# Discovering the files available

path = "../input/Kannada-MNIST/"

os.listdir(path)
# Importing the data

train = pd.read_csv(path+"train.csv")

test = pd.read_csv(path+"test.csv")

submission = pd.read_csv(path+"sample_submission.csv")
# Check the shape of our data

train.shape, test.shape
# Check the data samples

train.head(3)
# Check for any missing values

train.isnull().sum().sum(), test.isnull().sum().sum()
# Let us understand our target variable's distribution

sns.countplot(train["label"])

plt.show()
# Looking at individual samples

def visualize_image(ix=0):

    plt.imshow(train.iloc[ix, 1:].values.reshape(28, 28, 1)[:, :, 0])

    plt.title("Class = " + str(train["label"].loc[ix]))

    plt.show()



visualize_image(1)
# Setting the seed

seed = 10
# Splitting target and features

target = train["label"]

features = train.drop("label", 1)
from sklearn.decomposition import PCA

from umap import UMAP



pca = PCA(random_state=seed, n_components=50)

%time pca_features = pca.fit_transform(features)



umap = UMAP(n_neighbors=10, metric="cosine", random_state=seed, n_epochs=300)

%time umap_features = umap.fit_transform(pca_features)
plt.figure(figsize=(10, 7))

sns.scatterplot(umap_features[:, 0], umap_features[:, 1], hue=target, palette=sns.color_palette("Set1", target.nunique()))

plt.show()
# Splitting the data into train, val and test sets



x, x_val, y, y_val = train_test_split(umap_features, target, random_state=seed, stratify=target, test_size=0.2)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, stratify=y, test_size=0.2)
# Defining the classifiers

classifiers = {"LR": LogisticRegression(random_state=seed), "RF": RandomForestClassifier(random_state=seed), "KNN": KNeighborsClassifier(n_neighbors=20, n_jobs=-1)}
# Building the models

models = {}

val_preds = {}

test_preds = {}

for k, v in classifiers.items():

    print(f"{k}")

    %time models[k] = v.fit(x_train, y_train)

    val_preds[k] = models[k].predict(x_val)

    test_preds[k] = models[k].predict(x_test)

    print(f"Validation Accuracy: {np.round(models[k].score(x_val, y_val), 4)} | Test Accuracy: {np.round(models[k].score(x_test, y_test), 4)}")
# Analyzing the confusion matrix

plt.figure(figsize=(10, 7))

sns.heatmap(confusion_matrix(y_test, test_preds["KNN"]), annot=True, fmt='d')

plt.show()
# Plotting the decision boundary

from mlxtend.plotting import plot_decision_regions



plt.figure(figsize=(10, 7))

plot_decision_regions(x_test, y_test.values, clf=models["KNN"])

plt.show()
# Scaling the pixel values to a range between 0 and 1

X = features / 255

payload = test.drop("id", 1) / 255
# Reshaping the data into a 3 dimensional array

X = X.values.reshape(-1, 28, 28, 1)

payload = payload.values.reshape(-1, 28, 28, 1)



# Encoding the target variable. This is because we will be using softmax in the output layer and it outputs probabilities for each class

Y = tf.keras.utils.to_categorical(target, num_classes = 10)
# Splitting the data into train, val and test sets



x, x_val, y, y_val = train_test_split(X, target, random_state=seed, stratify=target, test_size=0.2)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=seed, stratify=y, test_size=0.2)
# Data augmentation

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1, # Randomly zoom image 

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False

)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation = tf.nn.relu, input_shape = (28, 28, 1)),

    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(10, activation = tf.nn.softmax)

])
optimizer=tf.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit_generator(data_gen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test), verbose=1, steps_per_epoch=x_train.shape[0]//64)
submission["label"] = np.argmax(model.predict(payload), 1)
submission.to_csv("submission.csv", index=False)
submission.head(3)
plt.imshow(test.iloc[1, 1:].values.reshape(28, 28, 1)[:, :, 0])
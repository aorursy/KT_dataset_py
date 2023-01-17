!pip install focal-loss
import tensorflow as tf
print("tensorflow version: " + tf.__version__)
from kaggle_datasets import KaggleDatasets
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications import DenseNet201
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.python.keras import backend
from focal_loss import BinaryFocalLoss
random_state = 19 # using a constant random seed makes the results more consistent and helps comparing between them.
# Get the path of the Current System (GCS)
GCS_PATH = KaggleDatasets().get_gcs_path("siim-isic-melanoma-classification")
# get the train data in dataframe format for quick examination of the data
dataframe_train = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")

dataframe_train.head(10)
# image_paths = GCS_PATH_TRAIN + "\\" + dataframe_train["image_name"]+".jpg" # \\ because \ is an escape character
# image_paths = "../input/siim-isic-melanoma-classification/jpeg/train/" + dataframe_train["image_name"] + ".jpg" 
image_paths = "../input/jpeg-melanoma-256x256/train/" + dataframe_train["image_name"] + ".jpg" 
f, ax = plt.subplots(3, 5, figsize = (10,6))
for i in range(15):
    #print(image_paths[i])
    img = cv2.imread(image_paths[i])
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # the default cv2 format is BGR
    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis("off")
plt.show()
    
dataframe_train["target"].value_counts()
downsampling = 1000
# sample 1000 benign samples and merge them together with all of the malignant samples
dataframe_train_benign_downsampled = dataframe_train[dataframe_train["target"]==0].sample(downsampling)
dataframe_train_malignant = dataframe_train[dataframe_train["target"]==1]
# join the two parts together. Note that now the two sample types are not mixed anymore in the data
# but appear in two blocks.
dataframe_train_downsampled = pd.concat([dataframe_train_benign_downsampled, dataframe_train_malignant])
image_paths = ["../input/jpeg-melanoma-256x256/train/" + dataframe_train_benign_downsampled["image_name"].values[i] + ".jpg" for i in range(downsampling)]
f, ax = plt.subplots(3, 5, figsize = (10,6))
for i in range(15):
    #print(image_paths[i])
    img = cv2.imread(image_paths[i])
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # the default cv2 format is BGR

    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis("off")
plt.show()
image_paths = ["../input/jpeg-melanoma-256x256/train/" + dataframe_train_malignant["image_name"].values[i] + ".jpg" for i in range(dataframe_train_malignant.shape[0])]

#print(image_paths[1])
f, ax = plt.subplots(3, 5, figsize = (10,6))
for i in range(15):
    #print(image_paths[i])
    img = cv2.imread(image_paths[i])
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # the default cv2 format is BGR
    ax[i//5, i%5].imshow(img)
    ax[i//5, i%5].axis("off")
plt.show()
image_paths = ["../input/jpeg-melanoma-256x256/train/" + dataframe_train_benign_downsampled["image_name"].values[i] + ".jpg" for i in range(downsampling)]

dataframe_train_labels = []
dataframe_train_images = []
for i in range(dataframe_train_downsampled.shape[0]):
    dataframe_train_labels.append(dataframe_train_downsampled["target"].values[i])
    dataframe_train_images.append("../input/jpeg-melanoma-256x256/train/" + dataframe_train_downsampled["image_name"].values[i] + ".jpg")
    
# create a dataframe from the columns
nparray_train_reduced_tuples = zip(dataframe_train_images, dataframe_train_labels)
# dataframe_train_reduced = pd.DataFrame(np.array([dataframe_train_labels, dataframe_train_images]), columns = ["label","image"])
dataframe_train_reduced = pd.DataFrame(nparray_train_reduced_tuples, columns = ["image","label"])
# dataframe_train_reduced = pd.DataFrame(np.array([dataframe_train_labels, dataframe_train_images]))
dataframe_train_reduced.head()
x_train, x_val, y_train, y_val = train_test_split(dataframe_train_reduced["image"], dataframe_train_reduced["label"],
                                                 test_size = 0.2, random_state = random_state)
dataframe_train_split = pd.DataFrame(zip(x_train,y_train), columns = ["image","label"])
dataframe_val = pd.DataFrame(zip(x_val,y_val), columns = ["image","label"])
gen_train = ImageDataGenerator(
    rescale = 1./255, # rescale the images (RGB [0,255])
    width_shift_range = 0.15, height_shift_range = 0.15, # randomly shift the pictures by 15% in both axes
    horizontal_flip = True, vertical_flip = True, # randomly flip the images in both axes
)
train_generator = gen_train.flow_from_dataframe(dataframe_train_reduced, x_col = "image", y_col = "label",
                                               target_size = (256,256), batch_size = 8,
                                               shuffle = True, # important as mentioned above
                                               class_mode = "raw")
val_generator = gen_train.flow_from_dataframe(dataframe_val, x_col = "image", y_col = "label",
                                               target_size = (256,256), batch_size = 8,
                                               shuffle = True, # not sure if important
                                               class_mode = "raw")

model = VGG16(weights = "imagenet",
             include_top = False, # because a new top will be added to match the dimensions of this dataset
             input_shape = (256,256,3))
x = Flatten()(model.output)
output = Dense(1,activation = "sigmoid")(x)
model = Model(model.input, output)


model.compile(loss = "binary_crossentropy", metrics = [tf.keras.metrics.AUC()], optimizer = Adam(lr=0.00001))
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = backend.binary_crossentropy(y_true, y_pred)
        
        y_pred = backend.clip(y_pred, backend.epsilon(), 1.- backend.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = backend.pow((1-p_t), gamma)

        # compute the final loss and return
        return backend.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
model.compile(loss=BinaryFocalLoss(gamma=2), metrics = [tf.keras.metrics.AUC()], optimizer = Adam(lr=0.00001))
model2 = DenseNet201(weights = "imagenet",
             include_top = False, # because a new top will be added to match the dimensions of this dataset
             input_shape = (256,256,3))
x = Flatten()(model2.output)
output = Dense(1,activation = "sigmoid")(x)
model2 = Model(model2.input, output)


model2.compile(loss=BinaryFocalLoss(gamma=2), metrics = [tf.keras.metrics.AUC()], optimizer = Adam(lr=0.00001))
batch_size = 8
steps_per_epoch = dataframe_train_reduced.shape[0] // batch_size
epochs = 3
validation_steps = dataframe_val.shape[0] // batch_size
history3 = model2.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs = epochs,
                    validation_data = val_generator, validation_steps = validation_steps)

predictions = [] # the test predictions will be stored here
# read the test csv file and obtain the image paths from it
dataframe_test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
test_image_paths = ["../input/jpeg-melanoma-256x256/test/" + image_name + ".jpg" for image_name in dataframe_test["image_name"]]
print(test_image_paths[5])
predictions = [] # the test predictions will be stored here
# read the test csv file and obtain the image paths from it
dataframe_test = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
test_image_paths = ["../input/jpeg-melanoma-256x256/test/" + image_name + ".jpg" for image_name in dataframe_test["image_name"]]
# go over the image paths, load their respective images, make a prediction for them and save the predictions
i=0
for test_image_path in test_image_paths:
    img = cv2.imread(test_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.reshape(img,(1,256,256,3))
    predictions.append(model.predict(img))
    if i%100 == 0:
        print("finished " + str(i) + " out of " + str(len(test_image_paths)))
    i += 1


submission = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
submission["target"] = predictions
submission.to_csv("submission.csv", index = False)

submission.head(30)
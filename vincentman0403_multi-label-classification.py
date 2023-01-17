from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import imutils
import numpy as np
import cv2
import os
from keras import preprocessing
import random
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="softmax"):
        model = Sequential()
        inputShape = (height, width, depth)

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model
def show_train_image(path):
    img = load_img(path)
    plt.figure(figsize=(8,8))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
# Show image in "black_jeans"
show_train_image('../input/dataset/black_jeans/00000000.jpg')
# Show image in "blue_dress"
show_train_image('../input/dataset/blue_dress/00000000.jpg')
# Show image in "blue_jeans"
show_train_image('../input/dataset/blue_jeans/00000000.jpg')
# Show image in "blue_shirt"
show_train_image('../input/dataset/blue_shirt/00000000.jpg')
# Show image in "red_dress"
show_train_image('../input/dataset/red_dress/00000000.jpg')
# Show image in "red_shirt"
show_train_image('../input/dataset/red_shirt/00000000.jpg')
# Convert data and labels as arrays 
imagePaths = sorted(list(paths.list_images("../input/dataset")))

# random shuffle
random.seed(42)
random.shuffle(imagePaths)

data = []
labels = []
image_dims = (96, 96, 3)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (image_dims[1], image_dims[0]))
    image = img_to_array(image)
    data.append(image)
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)
    
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("{} images ({:.2f}MB)".format(len(imagePaths), data.nbytes / (1024 * 1000.0)))    
data.shape
# Convert labels as sparse matrix
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# total 6 labels
print("class labels:")
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))
# Split data into train and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# Data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
model = SmallerVGGNet.build(
    width=image_dims[1], height=image_dims[0],
    depth=image_dims[2], classes=len(mlb.classes_),
    finalAct="sigmoid")
model.compile(loss="binary_crossentropy", optimizer='adam',
              metrics=["accuracy"])
batch_size = 32
epochs = 10
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=batch_size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs, verbose=1)
plt.figure(figsize=(10,10))
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
def predict(img_path):
    # load image and convert as array
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # classify the input image then find the indexes of the two class labels with the largest probability
    proba = model.predict(img)[0]
    idxs = np.argsort(proba)[::-1][:2]
    
    # show the probabilities for each of the individual labels
    for (label, p) in zip(mlb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))

    # plot image and label
    plt.figure(figsize=(10,10))
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
        plt.text(10, (i * 30) + 25, label, fontsize=16, color='y')

    output = load_img(img_path)    
    plt.xticks([])
    plt.yticks([])
    plt.imshow(output)
predict('../input/testset/black_jeans.jpg')
predict('../input/testset/blue_dress.jpg')
predict('../input/testset/blue_jeans.jpg')
predict('../input/testset/blue_shirt.jpg')
predict('../input/testset/red_dress.jpg')
predict('../input/testset/red_shirt.jpg')
predict('../input/testset/black_dress_01.jpg')
predict('../input/testset/black_dress_02.jpg')
predict('../input/testset/black_shirt_01.jpg')
predict('../input/testset/black_shirt_02.jpg')
predict('../input/testset/red_jeans_01.jpg')
predict('../input/testset/red_jeans_02.jpg')
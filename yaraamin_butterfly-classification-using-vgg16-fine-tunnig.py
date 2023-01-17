import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from os import listdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os

categories = []
# Setting variable filenames to path to iterate better 
filenames = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")
for filename in filenames:
        # Splitting the file.png to get the category 
        # Suppose /kaggle/input/butterfly-dataset/leedsbutterfly/images/001000.png
        category = filename.split(".")[0]
        # This will return 001000
        categories.append(category[0:3])
        # This will append the categories with 001
        
print(categories[0:5])
df = pd.DataFrame({
    "Image" : filenames,
    "Category" : categories
})
df.head()
df['Category'].value_counts()
df['Category'].value_counts().plot.bar()
data=[]
labels=[]
filenames = os.listdir("/kaggle/input/butterfly-dataset/leedsbutterfly/images/")
for filename in filenames:
    label = int(str(filename)[:3])
    image = cv2.imread('/kaggle/input/butterfly-dataset/leedsbutterfly/images/'+filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    labels.append(label)
data = np.array(data) /255

print(data[1].shape)
type(data)
data.shape
labels = np.array(labels)

labels = labels.reshape(len(labels),1)
print(labels.shape)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
labels = ohe.fit_transform(np.asarray(labels).reshape(len(labels),1))
type(labels)
data = np.array(data)
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.3, random_state=42)

trainAug = ImageDataGenerator(
    rotation_range=15,
    fill_mode="nearest")
baseModel = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(10, activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False
model.summary()

INIT_LR = 1e-3
EPOCHS = 25
BS = 8
opt = Adam(lr=INIT_LR)

model.compile(loss = 'categorical_crossentropy', optimizer = opt,
    metrics=["accuracy"])
# H = model.fit(trainX, trainY.toarray(), batch_size=BS, epochs = 30)
H = model.fit_generator(
    trainAug.flow(trainX, trainY.toarray(), batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY.toarray()),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)
predIdxs = model.predict(testX)


predIdxs = np.argmax(predIdxs, axis=1)
predIdxs
print(classification_report(testY.toarray().argmax(axis=1), predIdxs))

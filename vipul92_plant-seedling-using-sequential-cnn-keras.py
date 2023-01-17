import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from PIL import Image, ImageOps
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout,BatchNormalization

print(os.listdir("../input"))


#parsing all png files, and printing format, size and mode using PIL image module
import glob
train_imgs = []
train_label= []

train_dir = '../input/plant-seedlings-classification/train/*/*.png'

for img_dir in glob.glob(train_dir):
    img = Image.open(img_dir)
#     print("Label = " + img_dir.split('/')[-2] + " | for" + img_dir,img.format, img.size, img.mode)
#     print(img.resize((128, 128),Image.ANTIALIAS)) # ANTIALIAS to remove distortion, smoothening
    train_imgs.append(ImageOps.fit(img,(128, 128),Image.ANTIALIAS).convert('RGB'))
    train_label.append(img_dir.split('/')[-2])
images = np.array([np.array(im) for im in train_imgs])
images = images.reshape(images.shape[0], 128, 128, 3) / 255
lb = LabelBinarizer().fit(train_label)
label = lb.transform(train_label) 
trainX, validX, trainY, validY = train_test_split(images, label, test_size=0.05)
model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(128, 128, 3)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
##fitting our sequential model
model.fit(trainX, trainY,
          batch_size=64,
          epochs=50,#run with 50 epochs first to get 95% accuracy
          validation_split = 0.2)
test_dir = '../input/plant-seedlings-classification/test/*.png'
test_imgs=[]
names = []
for timage in glob.glob(test_dir):
    img = Image.open(timage)
    names.append(timage.split('/')[-1])
    test_imgs.append(ImageOps.fit(img,(128, 128),Image.ANTIALIAS).convert('RGB'))
    
test_images = np.array([np.array(im) for im in test_imgs])
test_images_X = test_images.reshape(test_images.shape[0], 128, 128, 3) / 255
test_y = lb.inverse_transform(model.predict(test_images_X))
df = pd.DataFrame(data={'file': names, 'species': test_y})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('resultss.csv', index=False)

print(df_sort)
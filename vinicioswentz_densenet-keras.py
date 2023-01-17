# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import cv2



from PIL import Image



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

print(os.listdir("../input"))



MAIN_FOLDER = os.listdir("../input/")



# Any results you write to the current directory are saved as output.

bee_data = pd.read_csv("../input/pollendataset/PollenDataset/pollen_data.csv")
a = Image.open("../input/pollendataset/PollenDataset/images/NP20833-61r.jpg")

print(a.size)

plt.imshow(a)
bee_data.head()
bee_data = bee_data.drop(["Unnamed: 0"], axis=1)
bee_data.head()
bee_data = bee_data.set_index(["filename"])
bee_data.loc["P10057-125r.jpg"][0]
images = glob.glob("../input/pollendataset/PollenDataset/images/*.jpg")
len(images)
bee_images = []

labels = []
for image in images:

    filename =  image.split("/")[-1]

    label = bee_data.loc[filename][0]

    opened_image = cv2.imread(image)

    array_image = Image.fromarray(opened_image, "RGB")

    resize_img = array_image.resize((180 , 180))

    rotated45 = resize_img.rotate(45)

    rotated75 = resize_img.rotate(75)

    blur = cv2.blur(np.array(resize_img) ,(10,10))

    #print("filename: {}, pollen carrying: {}".format(filename, label))

    bee_images.append(np.array(resize_img))

    bee_images.append(np.array(rotated45))

    bee_images.append(np.array(rotated75))

    bee_images.append(np.array(blur))

    

    labels.append(label)

    labels.append(label)

    labels.append(label)

    labels.append(label)
bees = np.array(bee_images)

labels = np.array(labels)



np.save('Bees' , bees)

np.save('Labels' , labels)



print(len(bees))

print(len(labels))
from sklearn.model_selection import train_test_split



train_x , x , train_y , y = train_test_split(bees , labels , test_size = 0.2 , random_state = 111)



eval_x , test_x , eval_y , test_y = train_test_split(x , y ,test_size = 0.5 , random_state = 111)
from tensorflow.keras.applications.densenet import DenseNet169, DenseNet201, DenseNet121

from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate

from tensorflow.keras.models import Model, load_model

from tensorflow.keras import losses, optimizers, activations, metrics
class Densenet:

    def __init__(self, loss, optmizer, metrics):

        self.loss = loss

        self.optimizer = optmizer

        self.metrics = metrics

    

    def create_model(self, output_space: int) -> Model:

        base_model = DenseNet121(input_shape=(180, 180, 3), include_top=False)

        out0 = base_model.output

        out1 = GlobalMaxPooling2D()(out0)

        out2 = GlobalAveragePooling2D()(out0)

        out3 = Flatten()(out0)

        out = Concatenate(axis=-1)([out1, out2, out3])

        out = Dropout(0.5)(out)

        

        predictions = Dense(output_space, activation="sigmoid")(out)

        

        model = Model(inputs=base_model.input, outputs=predictions)



        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        #model.summary()

        return model
from keras.callbacks import LearningRateScheduler

from keras.callbacks import ReduceLROnPlateau



def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.5

    epochs_drop = 2

    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

    return lrate



learning_rate = LearningRateScheduler(step_decay)



rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-7)



callbacks_list = [learning_rate, rlrp]
model = Densenet(

    losses.binary_crossentropy,

    optimizers.Adam(lr=10e-5),

    ["accuracy"]

).create_model(1)
model.fit(train_x, train_y,

          validation_data=(eval_x, eval_y),

          batch_size=32,

          epochs=30,

          verbose=1)
accuracy = model.evaluate(test_x, test_y, verbose=1)

print('\n', 'Test_Accuracy:-', accuracy[1])
results = model.predict(test_x)
results = np.argmax(results,axis=1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,2857),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)
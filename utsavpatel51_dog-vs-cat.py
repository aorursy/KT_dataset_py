import os

import cv2

import numpy as np

import pandas as pd

import seaborn

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from random import shuffle

from keras.models import Sequential

from sklearn.metrics import confusion_matrix,classification_report

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image
def plot_val_acc(history):

    fig,axs= plt.subplots(1,2,figsize=(10,5))

    axs[0].plot(history.history['acc'])

    axs[0].plot(history.history['val_acc'])

    axs[0].set_title('Model accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values

    axs[1].plot(history.history['loss'])

    axs[1].plot(history.history['val_loss'])

    axs[1].set_title('Model loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['Train', 'Test'], loc='upper left')

    plt.show()
X_train=[]

y_train=[]

filenames = os.listdir("../input/dogs-vs-cats/train/train")

for filename in filenames:

    label = filename.split('.')[0]

    if label=="cat":

        y_train.append(0)

    elif label=="dog":

        y_train.append(1)

    img_arr = cv2.imread("../input/dogs-vs-cats/train/train/"+filename, cv2.IMREAD_GRAYSCALE)

    img_arr = cv2.resize(img_arr, dsize=(128, 128))

    X_train.append(img_arr)
X_train = np.array(X_train)

X_train.shape
X_train = X_train/255

X_train = X_train.reshape(-1, 128, 128, 1)

X_train.shape
y_train1 = to_categorical(y_train)
def model_1():

    model = Sequential()

    model.add(Conv2D(32,(3,3),activation="relu",input_shape=(128,128,1)))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(64,(3,3),activation="relu"))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(128,(3,3),activation="relu"))

    model.add(MaxPooling2D(2,2))



    model.add(Flatten())

    model.add(Dense(512,activation="relu"))

    model.add(Dropout(0.5))

    

    model.add(Dense(256,activation="relu"))

    model.add(Dropout(0.5))

    

    model.add(Dense(2,activation="softmax"))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model
'''def with_aug():

    model1 = model_1()

    history1 = model1.fit(X_train,y_train1,batch_size=32,epochs=20,verbose=1,validation_split=0.25)

    return history1'''
'''history1 = with_aug()'''
'''plot_val_acc(history1)'''
aug = ImageDataGenerator()

print("[INFO] performing 'on the fly' data augmentation")

aug = ImageDataGenerator(

        rotation_range=20,

        zoom_range=0.15,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.15,

        horizontal_flip=True,

        fill_mode="nearest")
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
model2 = model_1()

trainX, testX, trainy, testy = train_test_split(X_train,y_train1,test_size=0.2,random_state=42)

histoty2 = model2.fit_generator(aug.flow(trainX, trainy, batch_size=24),

                            validation_data=(testX,testy),

                            steps_per_epoch=len(trainX) // 24,

                            epochs=40,callbacks=[es])
plot_val_acc(histoty2)
truey =[]

for i in testy:

    truey.append(0) if i[0]==1 else truey.append(1)
predict2 = model2.predict_classes(testX)

print(classification_report(predict2,truey))
print(confusion_matrix(predict2,truey))
X_test=[]

filenames = os.listdir("../input/dogs-vs-cats/test1/test1/")

for filename in filenames:

    img_arr = cv2.imread("../input/dogs-vs-cats/test1/test1/"+filename, cv2.IMREAD_GRAYSCALE)

    img_arr = cv2.resize(img_arr, dsize=(128, 128))

    X_test.append(img_arr)
X_test = np.array(X_test)

X_test = X_test/255.0

X_test = X_test.reshape(-1, 128, 128, 1)

X_test.shape
predict = model2.predict(X_test)
model2.predict_classes(X_test[0].reshape(-1, 128, 128, 1))
p = np.argmax(predict, axis=-1)
model1.predict_classes(X_test)
df = pd.DataFrame({"id":[i+1 for i in range(12500)], 

                   "lable" : p})

df.to_csv(filename, index=False)
fig,axs = plt.subplots(1,10,figsize=(15,18),sharey=True)

for i,filename in enumerate(filenames[:10]):

    img_arr = cv2.imread("../input/dogs-vs-cats/test1/test1/"+filename)

    img_arr = cv2.resize(img_arr, dsize=(128, 128))

    axs[i].imshow(img_arr)

    label = "dog" if p[i]==1 else "cat"

    axs[i].set_title(label)
'''model_json = model1.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

model1.save_weights("model.h5")'''
'''from IPython.display import FileLink

FileLink(r'model.h5')'''
'''model1.save("model.h5")

FileLink(r'model.h5')'''
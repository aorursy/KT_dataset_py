import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





from keras.models import load_model #eğitimi saklamak

from keras.callbacks import ReduceLROnPlateau  #eğitimi tekrar geri çağırmak için kullanılır. 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_train.tail()
x_train = df_train.drop(columns=['label'],axis = 1)

y_train = df_train["label"]
plt.figure(figsize=(15,7))

g = sns.countplot(y_train, palette="icefire")

plt.title("Number of digit classes")

y_train.value_counts()


from PIL import Image



# herhangi bir sayıyı yazdıran kod

img = x_train.iloc[0].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(df_train.iloc[0,0])

plt.axis("off")

plt.show()



# herhangi bir sayıyı yazdıran kod

img = x_train.iloc[8].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(df_train.iloc[8,0])

plt.axis("off")

plt.show()


from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



y_train = to_categorical(df_train["label"],num_classes = 10)

y_train
x_train = x_train / 255.0

print("x_train shape: ",x_train.shape)

x_train = x_train.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train.shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state=2)

print("x_train shape",x_train.shape)

print("x_test shape",x_test.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_test.shape)


from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator





input_shape = (28,28,1)

num_classes = 10



model = Sequential()

model.add(Conv2D(16, kernel_size = (3,3), activation = "relu", padding = "Same", input_shape = input_shape))

model.add(Conv2D(16, kernel_size = (3,3), activation = "relu", padding = "Same"))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(32, kernel_size = (2,2), activation = "relu", padding = "Same"))

model.add(Conv2D(32, kernel_size = (2,2), activation = "relu", padding = "Same"))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(num_classes,activation="softmax"))

model.summary()



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])



epochs = 100

batch_size = 1000
datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.5,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.5, # Randomly zoom image 5%

        width_shift_range=0.5,  # randomly shift images horizontally 5%

        height_shift_range=0.5,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

epochs = epochs, validation_data = (x_test,y_test), steps_per_epoch=x_train.shape[0] // batch_size)
plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
from sklearn.metrics import confusion_matrix





# Predict the values from the validation dataset

y_pred = model.predict(x_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
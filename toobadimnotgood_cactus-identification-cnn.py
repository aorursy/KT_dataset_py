import os



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#for image processing

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image



from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score





import matplotlib.pyplot as plt

%matplotlib inline



train_dir = "../input/train/train"

test_dir = "../input/test/test"


#training df import

df = pd.read_csv("../input/train.csv")

print(df.shape) #we have quite a bit of data, not too little.

#The dataset isn't balanced, yet it's not too heavily leaning toward 1 class or the other. A normal CNN could still have

#a decent result with this set, assuming that the differences will be distinct enough.

print(df.has_cactus.value_counts(normalize=True))

df.head()
#create a list to hold the 4d image tensors data. Note: the order of training images and df's labels are identical.

def load_imgs(train_dir,df):

    img_lst = []

    for img in df["id"]:

        with open(os.path.join(train_dir, img), 'rb') as i:

            img = image.load_img(i,target_size=(32,32))

            img = image.img_to_array(img)/255.0 #normalization

            img_lst.append(image.img_to_array(img))

    return img_lst





#return random indices from a df, particularly use to see random images

def random_imgs(df,num_images,train):

    index_lst = df["id"].sample(n=num_images).index

    img_lst = []

    for i in index_lst:

        img_lst.append(train[i])

    return img_lst



#only plot 2x2 images. Helper function. One can always generalize the function if neccessary

def plot_examples(img_lst,title):

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8,8))

    ax[0].imshow(img_lst[0])

    ax[0].set_title(title)

    ax[1].imshow(img_lst[1])

    #ax[1].set_title(title)

    ax[2].imshow(img_lst[2])

    #ax[2].set_title(title)

    ax[3].imshow(img_lst[3])

    #ax[3].set_title(title)

    plt.show()

#loading training data

train = load_imgs(train_dir,df)

print(np.shape(train))
#to pick random images for viewring from both labels (cactus, no cactus)

cactus = random_imgs(df[df["has_cactus"]==1],4,train)

no_cactus = random_imgs(df[df["has_cactus"] == 0],4,train)

plot_examples(cactus,"cactus")

plot_examples(no_cactus,"no cactus")
from sklearn.model_selection import train_test_split

#since we have the target and the imported tensors in order, it's quite easy for us to split our data.

#also, with the number of data we have, it's not too low such that we'll be utilising the k-fold val.



def train_val_split(train,df):

    #keeping random_state for consistency with future models for comparison

    X_train, X_val, y_train, y_val = train_test_split(train,df["has_cactus"], test_size=0.3,stratify=df["has_cactus"],random_state=1)

    return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)



X_train, X_val, y_train, y_val = train_val_split(train,df)


#we'll be using keras for prototyping first

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Activation

from keras.layers import BatchNormalization

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator #image augmentation

from keras import callbacks



history = callbacks.History() #need to be defined first
#using functional API

from keras import Input, layers, Model



#to combat overfitting, better optimization for CNN, we'll be using Batch normalization PRIOR to activation.

#There has been a debate on where to use it, but the consensus has been to use it prior/after non-linearity

def train_cnn():

    input_tensor = Input(shape=(32,32,3,)) #our image dimension. Unknown sample size input

    conv1 = layers.Conv2D(filters=9,kernel_size=(3,3))(input_tensor)

    conv1 = layers.BatchNormalization()(conv1)

    conv1 = layers.Activation("relu")(conv1)

    pool1 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv1)



    conv2 = layers.Conv2D(filters=17,kernel_size=(3,3))(pool1)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.Activation("relu")(conv2)

    pool2 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv2)



    conv3 = layers.Conv2D(filters=31,kernel_size=(3,3))(pool2)

    conv3 = layers.BatchNormalization()(conv3)

    conv3 = layers.Activation("relu")(conv3)

    pool3 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv3)



    flat = layers.Flatten()(pool3)

    flat = layers.Dropout(rate=0.3)(flat) #for hidden layer

    hidden = layers.Dense(units=500,activation="relu")(flat)

    output_tensor = layers.Dense(units=1,activation="sigmoid")(hidden)



    model = Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=optimizers.rmsprop(lr=0.0001), loss="binary_crossentropy",metrics=["accuracy"])

    print(model.summary())

    

    return model
model1 = train_cnn()

history1 = model1.fit(X_train,y_train,validation_data=(X_val,y_val),

                      verbose=True,shuffle=True,epochs=50)
def model_plot(history,epochs,title,y_range=[0.8,1.0],save=0 ):

    train_losses = history.history["loss"]

    val_losses = history.history["val_loss"]

    plt.plot([i for i in range(0,epochs)],train_losses,val_losses)

    plt.legend(["Train Loss","Val Loss"])

    plt.title(title)

    

    if save == 1:

        plt.savefig(title+"_Losses.jpg",dpi=1000)

    plt.show()

    

    

    train_losses = history.history["acc"]

    val_losses = history.history["val_acc"]

    plt.plot([i for i in range(0,epochs)],train_losses,val_losses)

    plt.legend(["Train_acc","Val_acc"])

    plt.title(title)

    plt.ylim(y_range)

    

    if save == 1:

        plt.savefig(title+"_Accuracy.jpg")

    plt.show()
#uncomment for plotting

model_plot(history1,epochs=50,title="baseline_cnn")


#Data Augmentation. Generating additional data for training

#refer to Keras for extra documentation as well as

#https://machinelearningmastery.com/image-augmentation-deep-learning-keras/ for a brief introduction

def data_aug(X_train):

    datagen = ImageDataGenerator(

        rotation_range=20,

        shear_range=0.1,

        zoom_range=0.1,

        horizontal_flip=True,

        vertical_flip=True,

        width_shift_range=0.15,

        height_shift_range=0.15)

    

    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:

    return datagen



datagen = data_aug(X_train)
model2 = train_cnn()

history2 = model2.fit_generator(datagen.flow(np.array(X_train),np.array(y_train),batch_size=30),

                              validation_data= (np.array(X_val),np.array(y_val)),

                              steps_per_epoch=len(X_train) / 50,epochs=50)
#uncomment for plotting

model_plot(history2,epochs=50,title="cnn_augmented")
from keras.applications import VGG16



#refer to: https://keras.io/applications/#vgg16

#we just take the conv base and frozen it, since it should help with broad classification

conv_base = VGG16(weights='imagenet',include_top=False, input_shape=(32,32,3))

conv_base.summary()
#shouldn't we do a baseline prediction with the VGG16 pre-trained network by itself first?

conv_base.trainable=False #we don't want to touch the generality of our VGG16

#encapsulating for later usage

def vgg_layer(conv_base):

    conv_base.trainable=False

    model3 = Sequential()

    model3.add(conv_base) #output = 512 as in summary above, treat as a "layer".

    model3.add(Flatten())

    model3.add(Dense(400))

    

    #tested these out already

    #model3.add(Activation("relu"))

    #model3.add(Dropout(rate=0.3))

    #model3.add(Dense(1,activation="sigmoid")) #output

    return model3

input_tensor = Input(shape=(32,32,3)) #similar to train_cnn() defined above

vgg = vgg_layer(conv_base)(input_tensor)

activation = layers.Activation("relu")(vgg)

dropout = layers.Dropout(rate=0.3)(activation)

output_tensor = layers.Dense(1, activation="sigmoid")(dropout)



model3 = Model(inputs=input_tensor,outputs=output_tensor)

model3.compile(optimizer=optimizers.rmsprop(lr=0.0001), loss="binary_crossentropy",metrics=["accuracy"])
history3 = model3.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=30,

                      verbose=True,shuffle=True,epochs=50)
model_plot(history3,epochs=50,title="VGG_Conv_Base")
#there are other inception models and there are pre-made versions in Keras, but we can create one ourselves here using VGG + the first model.

#the output of VGG is 512 (flattened). It's what we had defined in the vgg_layer() function above.

def inception_cnn():

    input_tensor = Input(shape=(32,32,3,)) #our image dimension. Unknown sample size input

    conv1 = layers.Conv2D(filters=9,kernel_size=(3,3))(input_tensor)

    conv1 = layers.BatchNormalization()(conv1)

    conv1 = layers.Activation("relu")(conv1)

    pool1 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv1)



    conv2 = layers.Conv2D(filters=17,kernel_size=(3,3))(pool1)

    conv2 = layers.BatchNormalization()(conv2)

    conv2 = layers.Activation("relu")(conv2)

    pool2 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv2)



    conv3 = layers.Conv2D(filters=31,kernel_size=(3,3))(pool2)

    conv3 = layers.BatchNormalization()(conv3)

    conv3 = layers.Activation("relu")(conv3)

    pool3 = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(conv3)



    flat = layers.Flatten()(pool3)

    flat = layers.Dropout(rate=0.3)(flat) #for hidden layer

    hidden = layers.Dense(units=500)(flat)

    

    #VGG16 up to the dense layer

    vgg = vgg_layer(conv_base)(input_tensor)

    

    

    #a combination of our own cnn + vgg + 450 dense with lr=0.0001 brought the accuracy up to low ~0.99.

    #let's see if re-using the original input will help with the vanishing gradient problem, and overfitting.

    #we'll be using depthwise separable convolution, point-wise convolution of 1x1/mixing channels while not touching the spatial aspect (conv)

    #https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

    

    separable  = layers.SeparableConv2D(filters=11,kernel_size=(1,1),activation="relu")(input_tensor)

    pool_separable = layers.MaxPool2D(pool_size=(2,2),padding="SAME")(separable)

    flat_separable = layers.Flatten()(pool_separable)

    

    #hidden is the original cnn dense layer, vgg is the vgg's dense layer. Combining the input for better details/broad analysis

    combined = layers.concatenate([hidden,vgg,flat_separable],axis=-1) #list input

    final_dense = layers.Dense(450)(combined)

    activation = layers.Activation("relu")(final_dense)

    dropout = layers.Dropout(rate=0.25)(activation)

    output_tensor = layers.Dense(1, activation="sigmoid")(dropout)

    

    

    model = Model(inputs=input_tensor, outputs=output_tensor)

    print(model.summary())

    model.compile(optimizer=optimizers.rmsprop(lr=0.0001), loss="binary_crossentropy",metrics=["accuracy"])

    print(model.summary())

    return model
model4 = inception_cnn()

history4 = model4.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=30,

                      verbose=True,shuffle=True,epochs=50)
model_plot(history4,epochs=50,title="Inception_example")
model1.save_weights("model1.h5",overwrite=False)

model2.save_weights("model2.h5",overwrite=False)

model3.save_weights("model3.h5",overwrite=False)

model4.save_weights("model4.h5",overwrite=False)
from keras.callbacks import ModelCheckpoint,TensorBoard 

import tensorflow as tf
#you may need to install the h5py library to output network weights in HDF5 format.

#refer to: https://machinelearningmastery.com/check-point-deep-learning-models-keras/ for information on how to use checkpoint (personal)



filepath = "best_weight.hdf5" #model checkpoints will be saved with the epoch number and the validation accuracy in the filename.

checkpointer = ModelCheckpoint(filepath=filepath,monitor="val_acc", verbose=1, save_best_only=True,save_weights_only=False,mode="max",period=1) #max because we want max accuracy



my_log_dir = "./logs"

tensorboard = TensorBoard(log_dir=my_log_dir, histogram_freq=0, batch_size=30, write_graph=True) #just need the architecture



history4 = model4.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=30,callbacks=[checkpointer,tensorboard],

                      verbose=True,shuffle=True,epochs=20)
%load_ext tensorboard.notebook

%tensorboard --logdir=my_log_dir
model4.load_weights("best_weight.hdf5")
df_test = pd.read_csv("../input/sample_submission.csv")

df_test.head()
test = load_imgs(test_dir,df_test)
pred = model4.predict(np.array(test))
df_test['has_cactus'] = pred

df_test.to_csv("submission.csv",index=False)
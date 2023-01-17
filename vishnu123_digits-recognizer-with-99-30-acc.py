import tensorflow as tf

from tensorflow.keras.layers import Conv2D,Activation,Lambda,BatchNormalization,Dropout,Flatten,MaxPooling2D,Input,Dense,MaxPool2D

from tensorflow.keras.models import Model,Sequential

from tensorflow.keras.datasets import mnist

from tensorflow.keras.utils import plot_model

from tensorflow.keras.losses import binary_crossentropy,mse

from tensorflow.keras import backend as K 

from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array

from tensorflow.keras.utils import to_categorical



from tensorflow.keras.callbacks import (

    ReduceLROnPlateau,

    EarlyStopping,

    ModelCheckpoint,

    TensorBoard

)



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm

import seaborn as sns
train = pd.read_csv("../input/digit-recognizer/train.csv")

sample,features = train.shape[0],train.shape[1]

print(f"Train data has {sample} rows and {features} columns")

train.head()
test = pd.read_csv("../input/digit-recognizer/test.csv")

sample,features = test.shape[0],test.shape[1]

print(f"Test data has {sample} rows and {features} columns")

test.head()
# let's see the distribution of classes

plt.xkcd()

plt.figure(figsize = (16,12))

ax = sns.barplot(x = train['label'].value_counts().index,y = train['label'].value_counts().values)



rects = ax.patches



for rect in rects:

    height = rect.get_height()

    ax.text(rect.get_x() + 1,height + 20, str (np.round(height,0)) + " ( " + str (np.round(height/60000,4)) + " )", ha = 'center',va = 'center_baseline')







plt.xlabel("Handwritten Digits",fontsize = 20)

plt.ylabel("Frequency",fontsize = 20)

plt.show()
# how our input data looks

from matplotlib import pyplot

plt.figure(figsize = (16,12))

for i in range(0, 9):

    pyplot.subplot(330 + 1 + i)

    r = np.random.choice(10000)

    pyplot.imshow(train.drop(['label'],axis=1).iloc[r].values.reshape(28,28), cmap=pyplot.get_cmap('gray'))

    plt.grid(False)

    plt.axis(False)

plt.show()
def generate_data(df,label = 0):

    X = train[train['label']==label]

    datagen = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,

                                zca_whitening=False,rotation_range = 20,

                                width_shift_range = 0.2,height_shift_range = 0.2,

                                shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,

                                vertical_flip = True,fill_mode = 'nearest'

                            )

    X = X.drop(['label'],axis=1).values.reshape((X.shape[0], 28, 28, 1))

    datagen.fit(X)

    for x_batch,y_batch in datagen.flow(X,[label]*len(X),batch_size = 9):

        plt.figure(figsize = (16,12))

        for i in range(0,9):

            pyplot.subplot(330 + 1 + i)

            pyplot.imshow(X[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))

        # show the plot

        pyplot.show()

        break
## let's generate augmented image of label 5

generate_data(train,label = 5)

# same can be done using above function for label between 0 to 9

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
# CREATE MORE IMAGES VIA DATA AUGMENTATION

datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
# Split the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42,stratify = Y_train)
def ConvNet(x,filters,size,strides=1,batch_norm = True):

    if strides==1:

        padding = 'same'

    else:

        padding = 'valid'

    x = Conv2D(filters = filters,kernel_size = size,strides = strides,padding = padding

              ,use_bias = not batch_norm,kernel_regularizer = tf.keras.regularizers.l2(0.005))(x)

    if batch_norm:

        x = BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    return(x)



def NetResidual(x,filters):

    prev = x

    x = ConvNet(x,filters//2,1)

    x = ConvNet(x,filters,3)

    x = tf.keras.layers.Add()([prev,x])

    return(x)



def NetBlock(x,filters,blocks):

    x = ConvNet(x,filters,3,strides=2)

    for _ in range(blocks):

        x = NetResidual(x,filters)

    return(x)

        

def Network(filters,x_input,name = 'Res_Con'):

    x = inputs = Input(x_input.shape[1:])

    x = ConvNet(x,32,3)

    x = NetBlock(x,64,3)

    x = Flatten()(x)

    x = Dense(256,activation = 'relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(10,activation = 'softmax')(x)

    return(Model(inputs,x,name = name))
model1 = Network(32,X_train)

model1.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

ckpt = ModelCheckpoint('res_model.h5',

                            verbose=1, save_weights_only=True,save_best_only = True)

epochs = 50

batch_size = 86



model1.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
history1 = model1.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = 30, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction,ckpt])



#model1.load_weights("../input/digit-recognizer-model/res_model.h5")
plt.figure(figsize = (16,8))

plt.subplot(1,2,1)

plt.plot(history1.history['loss'], color='b', label="Training loss")

plt.plot(history1.history['val_loss'], color='r', label="validation loss",)

plt.legend(loc='best', shadow=True)

plt.subplot(1,2,2)

plt.plot(history1.history['accuracy'], color='b', label="Training accuracy")

plt.plot(history1.history['val_accuracy'], color='r',label="Validation accuracy")

plt.legend(loc='best', shadow=True)

plt.suptitle("Model 1 Performance",fontsize = 30)

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                              cmap=plt.cm.Paired):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(16,8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="red" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model1.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model2 = Sequential()



model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model2.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model2.add(MaxPool2D(pool_size=(2,2)))

model2.add(Dropout(0.25))





model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model2.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model2.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model2.add(Dropout(0.25))





model2.add(Flatten())

model2.add(Dense(256, activation = "relu"))

model2.add(Dropout(0.5))

model2.add(Dense(10, activation = "softmax"))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

ckpt = ModelCheckpoint('cnn_model.h5',

                            verbose=1, save_weights_only=True,save_best_only = True)

epochs = 30 

batch_size = 86



model2.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
history2 = model2.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = 30, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction,ckpt])
plt.figure(figsize = (16,8))

plt.subplot(1,2,1)

plt.plot(history2.history['loss'], color='b', label="Training loss")

plt.plot(history2.history['val_loss'], color='r', label="validation loss",)

plt.legend(loc='best', shadow=True)

plt.subplot(1,2,2)

plt.plot(history2.history['accuracy'], color='b', label="Training accuracy")

plt.plot(history2.history['val_accuracy'], color='r',label="Validation accuracy")

plt.legend(loc='best', shadow=True)

plt.suptitle("Model 2 Performance",fontsize = 30)

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                              cmap=plt.cm.coolwarm_r):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize=(16,8))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="red" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model1.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

def predict_test(data,model):

    data = data / 255.0

    data = data.values.reshape(-1,28,28,1)

    return(model.predict(data))



res = predict_test(test,model1)
final_result = []

for i in range(len(res)):

    final_result.append(np.argmax(res[i]))

sub['Label'] = final_result
# skip connection network

model1.load_weights("../input/digit-recognizer-model/res_model.h5")



# cnn model

model2.load_weights("../input/digit-recognizer-model/cnn_model.h5")



def display(mode1,model2,data):

    # (batch_size,img_dim,img_dim,filters)

    # model 1 prediction

    p1 = np.argmax(model1.predict(data.values.reshape(-1,28,28,1)),axis=1)

    # model 2 prediction

    p2 = model2.predict_classes(data.values.reshape(-1,28,28,1))

    plt.figure(figsize = (24,10))

    for i in range(8):

        plt.subplot(2,4,i+1)

        r = np.random.choice(20000)

        plt.title(f"Model1 Predicted {p1[r]}\nModel2 Predicted {p2[r]}")

        plt.imshow(data.iloc[r].values.reshape(28,28),cmap = 'gray')

        plt.grid(False)

        plt.axis(False)

    plt.show()

        

    
display(model1,model2,test)
def error(model,data):

    try:

        # this is for funtional model

        p1 = np.argmax(model.predict(data.drop(['label'],axis=1).values.reshape(-1,28,28,1)),axis=1)

    except:

        # this is for sequential model

        p1 = model.predict_classes(data.drop(['label'],axis=1).values.reshape(-1,28,28,1))

    

    df = data[p1-data['label'].values!=0]

    plt.figure(figsize = (24,10))

    i=0

    for indx in np.random.choice(df.index.tolist(),8):       

        plt.subplot(2,4,i+1)

        #r = np.random.choice(20000)

        plt.title(f"Actual Label {data.iloc[indx]['label']}\nModel1 Predicted {p1[indx]}")

        plt.imshow(train.drop(['label'],axis=1).iloc[indx].values.reshape(28,28),cmap = 'gray')

        plt.grid(False)

        plt.axis(False)

        i+=1

    plt.show()
error(model1,train)
error(model2,train)
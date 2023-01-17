! swapon -s

! dd if=/dev/zero of=/swapfile bs=1024 count=1024k

! mkswap /swapfile

! swapon /swapfile
import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

import fnmatch

from glob import glob

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras.utils import to_categorical

import cv2

from sklearn.model_selection import train_test_split

from IPython.display import Image

from sklearn.metrics import confusion_matrix

np.random.seed(1)

tf.random.set_random_seed(11)
imagePatches = glob('../input/chest_xray/chest_xray/**/**/*.jpeg', recursive=True)
pattern_normal = '*NORMAL*'

pattern_bacteria = '*_bacteria_*'

pattern_virus = '*_virus_*'



normal = fnmatch.filter(imagePatches, pattern_normal)

bacteria = fnmatch.filter(imagePatches, pattern_bacteria)

virus = fnmatch.filter(imagePatches, pattern_virus)

x = []

y = []

for img in imagePatches:

    full_size_image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    im = cv2.resize(full_size_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    x.append(im)

    if img in normal:

        y.append(0)

    elif img in bacteria:

        y.append(1)

    elif img in virus:

        y.append(2)

    else:

        #break

        print('no class')

x = np.array(x)

y = np.array(y)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state = 1)

y_train = to_categorical(y_train, num_classes = 3)

y_valid = to_categorical(y_valid, num_classes = 3)

del x, y
x_train=x_train.reshape((x_train.shape[0], 224, 224,1))/255

x_valid=x_valid.reshape((x_valid.shape[0], 224, 224,1))/255

x_train.shape,x_valid.shape
def translate_label(inp):

    if inp[0]==1:

        return 'normal'

    elif inp[1]==1:

        return 'bacteria'

    else:

        return 'virus'

def translate_predicted_label(inp):

    maxim=np.argmax(inp)

    if maxim==0:

        return 'normal'

    elif maxim==1:

        return 'bacteria'

    else:

        return 'virus'
w,h=6,6

f, ax = plt.subplots(h,w, figsize=(65,65))

for i in range(h):

    for j in range(w):

        ax[i][j].imshow(np.squeeze(x_valid[i*w+j].astype('float32')), cmap='gray')

        ax[i][j].set_title(translate_label(y_valid[i*w+j]))

plt.show()
class MyBlock(tf.keras.Model):

    def __init__(self, filters,pooling=True,batchnorm=True,dropout=None, **kwargs):

        super(MyBlock, self).__init__(**kwargs)

        self.dropout=dropout

        self.pooling=pooling

        self.batchnorm=batchnorm

        self.convlayer7_0=layers.Conv2D(filters,(7,7),padding='same',activation='relu')

        self.convlayer7_1=layers.Conv2D(filters,(7,7),padding='same',activation='relu') 

        self.convlayer5_0=layers.Conv2D(filters,(5,5),padding='same',activation='relu')

        self.convlayer5_1=layers.Conv2D(filters,(5,5),padding='same',activation='relu')    

#         self.convlayer3_0=layers.Conv2D(filters,(3,3),padding='same',activation='relu')

#         self.convlayer3_1=layers.Conv2D(filters,(3,3),padding='same',activation='relu') 

#         self.convlayer1_0=layers.Conv2D(int(filters/2),(1,1),padding='same',activation='relu')

#         self.convlayer1_1=layers.Conv2D(int(filters/2),(1,1),padding='same',activation='relu') 

        self.poolinglayer=layers.MaxPool2D(2)

        self.batchnormlayer=layers.BatchNormalization()

        self.concnetratelayer=layers.Concatenate()

        if self.dropout is not None:

            self.dropoutlayer=layers.Dropout(self.dropout)

    

    def call(self, input_tensor, training=False):

        x=input_tensor

#         x1=self.convlayer1_0(x)

#         x1=self.convlayer1_1(x1)       

#         x3=self.convlayer3_0(x)

#         x3=self.convlayer3_1(x3) 

        x5=self.convlayer5_0(x)

        x5=self.convlayer5_1(x5)

        x7=self.convlayer7_0(x)

        x7=self.convlayer7_1(x7)

        x=self.concnetratelayer([x5,x7])

        if self.dropout is not None:

            x=self.dropoutlayer(x)

        if self.batchnorm:

            x=self.batchnormlayer(x)

        if self.pooling:

            x=self.poolinglayer(x)

        return x
def get_model():

    model=tf.keras.Sequential()

    model.add(layers.InputLayer(input_shape=(224, 224, 1)))

    model.add(MyBlock(32))

    model.add(MyBlock(64))

    model.add(MyBlock(64))

    model.add(MyBlock(128,dropout=0.1))

    model.add(MyBlock(128,dropout=0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(512,activation='relu'))

    model.add(layers.Dense(3,activation='softmax'))

    model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    tf.keras.utils.plot_model(model, 'model.png',show_shapes=True)  

    return model

 



model=get_model()

model.summary()

Image('model.png')
callbacks = [

  tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss',restore_best_weights=True),

]

model.fit(x_train,y_train,epochs=50,batch_size=64,shuffle=True,validation_data=(x_valid,y_valid),callbacks=callbacks)
model.evaluate(x_valid,y_valid)


w,h=6,6

f, ax = plt.subplots(h,w, figsize=(65,65))

for i in range(h):

    for  j in range(w):

        ax[i][j].imshow(np.squeeze(x_valid[i*w+j].astype('float32')), cmap='gray')

        ax[i][j].set_title("True: "+translate_label(y_valid[i*w+j])+" .Predicted: "+translate_predicted_label(model.predict(np.asarray([x_valid[i*h+j]]))))

plt.show()
sns.heatmap(

    confusion_matrix(np.argmax(y_valid,axis=1),np.argmax(model.predict(x_valid),axis=1)),

    xticklabels=["normal", "bacteria", "virus"],

    yticklabels=["normal", "bacteria", "virus"],

    square=True,annot=True

)
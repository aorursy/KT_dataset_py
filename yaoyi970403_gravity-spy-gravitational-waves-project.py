from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, Model, Sequential

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

import tensorflow as tf

import json

import os
df = pd.read_csv("../input/gravity-spy-gravitational-waves/trainingset_v1d1_metadata.csv") 

df.label.value_counts()
im_height = 224

im_width = 224

batch_size = 256

epochs = 10
# create direction for saving weights

if not os.path.exists("save_weights"):

    os.makedirs("save_weights")
image_path = "../input/gravity-spy-gravitational-waves/"

train_dir = image_path + "train/train"

validation_dir = image_path + "validation/validation"

test_dir = image_path + "test/test"



train_image_generator = ImageDataGenerator( rescale=1./255,                        

                                            shear_range=0.2,

                                            zoom_range=0.2,

                                            horizontal_flip=True)



train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,

                                                           batch_size=batch_size,

                                                           shuffle=True,

                                                           target_size=(im_height, im_width),

                                                           class_mode='categorical')

    

total_train = train_data_gen.n





validation_image_generator = ImageDataGenerator(rescale=1./255)



val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,

                                                              batch_size=batch_size,

                                                              shuffle=False,

                                                              target_size=(im_height, im_width),

                                                              class_mode='categorical')

    

total_val = val_data_gen.n







test_image_generator = ImageDataGenerator(rescale=1./255)



test_data_gen = test_image_generator.flow_from_directory( directory=test_dir,

                                                          batch_size=batch_size,

                                                          shuffle=False,

                                                          target_size=(im_height, im_width),

                                                          class_mode='categorical')

    

total_test = test_data_gen.n
covn_base = tf.keras.applications.DenseNet201(weights='imagenet', include_top = False,input_shape=(im_height,im_width,3))

covn_base.trainable = True



print(len(covn_base.layers))



for layers in covn_base.layers[:-30]:

    layers.trainable = False



model = tf.keras.Sequential([

        covn_base,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(22, activation='softmax')

    ])

model.summary()     

model.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),

    loss = 'categorical_crossentropy',

    metrics=['accuracy']

)
reduce_lr = ReduceLROnPlateau(

                                monitor='val_loss', 

                                factor=0.1, 

                                patience=2, 

                                mode='auto',

                                verbose=1

                             )





checkpoint = ModelCheckpoint(

                                filepath='./save_weights/DenseNet201.ckpt',

                                monitor='val_acc', 

                                save_weights_only=False, 

                                save_best_only=True, 

                                mode='auto',

                                period=1

                            )



history = model.fit(x=train_data_gen,

                    steps_per_epoch=total_train // batch_size,

                    epochs=epochs,

                    validation_data=val_data_gen,

                    validation_steps=total_val // batch_size,

                    callbacks=[checkpoint, reduce_lr])
model.save_weights('./save_weights/DenseNet201.ckpt',save_format='tf')
# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_accuracy"]



# figure 1

plt.figure()

plt.plot(range(epochs), train_loss, label='train_loss')

plt.plot(range(epochs), val_loss, label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')



# figure 2

plt.figure()

plt.plot(range(epochs), train_accuracy, label='train_accuracy')

plt.plot(range(epochs), val_accuracy, label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
scores = model.evaluate(test_data_gen, verbose=1)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
from sklearn.metrics import confusion_matrix

import numpy as np  

import itertools
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm)) #???????????????

    misclass = 1 - accuracy #???????????????

    if cmap is None:

        cmap = plt.get_cmap('Blues') #?????????????????????

    plt.figure(figsize=(14, 10)) #??????????????????

    plt.imshow(cm, interpolation='nearest', cmap=cmap) #????????????

    plt.title(title) #????????????

    plt.colorbar() #???????????????



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=90) #x??????????????????45???

        plt.yticks(tick_marks, target_names) #y??????



    if normalize:

        cm = cm.astype('float32') / cm.sum(axis=1)

        cm = np.round(cm,2) #???????????????????????????

        



    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])): #???cm.shape[0]???cm.shape[1]?????????????????????????????????????????????????????????

        if normalize: #?????????

            plt.text(j, i, "{:0.2f}".format(cm[i, j]), #??????????????????

                     horizontalalignment="center",  #?????????????????????

                     color="white" if cm[i, j] > thresh else "black")  #??????????????????

        else:  #????????????

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",  #?????????????????????

                     color="white" if cm[i, j] > thresh else "black") #??????????????????



    plt.tight_layout() #????????????????????????,??????????????????????????????

    plt.ylabel('True label') #y??????????????????

    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass)) #x??????????????????

    plt.show() #????????????



class_indices = train_data_gen.class_indices 

labels = []

for key, val in class_indices.items():

    labels.append(key)



# ????????????????????????????????????

Y_pred = model.predict_generator(test_data_gen, total_test // batch_size + 1)

# ???????????????????????????one hit??????

Y_pred_classes = np.argmax(Y_pred, axis = 1)

# ??????????????????

confusion_mtx = confusion_matrix(y_true = test_data_gen.classes,y_pred = Y_pred_classes)

# ??????????????????

plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)
from PIL import Image

import numpy as np
#??????????????????????????????

class_indices = train_data_gen.class_indices 

#???????????????????????????????????????

inverse_dict = dict((val, key) for key, val in class_indices.items()) 
#??????????????????

img = Image.open("../input/gravity-spy-gravitational-waves/test/test/1080Lines/H1_2Xn58S1QCj_spectrogram_0.5.png")

# ?????????resize???224x224??????

img = img.resize((im_width, im_height))

img = img.convert("RGB")
# ?????????

img1 = np.array(img) / 255.

# ?????????????????????????????????????????????????????????

img1 = (np.expand_dims(img1, 0))

#?????????????????????????????????

result = np.squeeze(model.predict(img1))

predict_class = np.argmax(result)

print(inverse_dict[int(predict_class)],result[predict_class])

#???????????????????????????????????????

plt.title([inverse_dict[int(predict_class)],result[predict_class]])

#????????????

plt.imshow(img)
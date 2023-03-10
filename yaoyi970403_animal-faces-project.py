from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers, models, Model, Sequential

import matplotlib.pyplot as plt

import pandas as pd

from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,ModelCheckpoint

import tensorflow as tf

import json

import os
im_height = 299

im_width = 299

batch_size = 256

epochs = 15
# create direction for saving weights

if not os.path.exists("save_weights"):

    os.makedirs("save_weights")
image_path = "../input/animal-faces/afhq/"

train_dir = image_path + "train"

validation_dir = image_path + "val"



train_image_generator = ImageDataGenerator( rescale=1./255, 

                                            rotation_range=40, 

                                            width_shift_range=0.2,

                                            height_shift_range=0.2, 

                                            shear_range=0.2,

                                            zoom_range=0.2,

                                            horizontal_flip=True, 

                                            fill_mode='nearest')



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
covn_base = tf.keras.applications.Xception(weights='imagenet', include_top = False,input_shape=(im_height,im_width,3))

covn_base.trainable = True



print(len(covn_base.layers))



for layers in covn_base.layers[:-32]:

    layers.trainable = False



model = tf.keras.Sequential([

        covn_base,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(3, activation='softmax')

    ])

model.summary()     

model.compile(

    optimizer=tf.keras.optimizers.Adam(),

    loss = 'categorical_crossentropy',

    metrics=['accuracy']

)
def lrfn(epoch):

    LR_START = 0.00001

    LR_MAX = 0.0004

    LR_MIN = 0.00001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr



rng = [i for i in range(epochs)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)





checkpoint = ModelCheckpoint(

                                filepath='./save_weights/myXception.ckpt',

                                monitor='val_acc', 

                                save_weights_only=False, 

                                save_best_only=True, 

                                mode='auto',

                                period=1

                            )



history = model.fit(x=train_data_gen,

                    steps_per_epoch=total_train // batch_size,

                    epochs=5,

                    validation_data=val_data_gen,

                    validation_steps=total_val // batch_size,

                    callbacks=[checkpoint, lr_schedule])
model.save_weights('./save_weights/myXception.ckpt',save_format='tf')
# plot loss and accuracy image

history_dict = history.history

train_loss = history_dict["loss"]

train_accuracy = history_dict["accuracy"]

val_loss = history_dict["val_loss"]

val_accuracy = history_dict["val_accuracy"]



# figure 1

plt.figure()

plt.plot(range(5), train_loss, label='train_loss')

plt.plot(range(5), val_loss, label='val_loss')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('loss')



# figure 2

plt.figure()

plt.plot(range(5), train_accuracy, label='train_accuracy')

plt.plot(range(5), val_accuracy, label='val_accuracy')

plt.legend()

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.show()
from sklearn.metrics import confusion_matrix

import numpy as np  

import itertools
def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None,normalize=False):

    accuracy = np.trace(cm) / float(np.sum(cm)) #???????????????

    misclass = 1 - accuracy #???????????????

    if cmap is None:

        cmap = plt.get_cmap('Blues') #?????????????????????

    plt.figure(figsize=(10, 8)) #??????????????????

    plt.imshow(cm, interpolation='nearest', cmap=cmap) #????????????

    plt.title(title) #????????????

    plt.colorbar() #???????????????



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45) #x??????????????????45???

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

Y_pred = model.predict_generator(val_data_gen, total_val // batch_size + 1)

# ???????????????????????????one hit??????

Y_pred_classes = np.argmax(Y_pred, axis = 1)

# ??????????????????

confusion_mtx = confusion_matrix(y_true = val_data_gen.classes,y_pred = Y_pred_classes)

# ??????????????????

plot_confusion_matrix(confusion_mtx, normalize=True, target_names=labels)
from PIL import Image

import numpy as np
#??????????????????????????????

class_indices = train_data_gen.class_indices 

#???????????????????????????????????????

inverse_dict = dict((val, key) for key, val in class_indices.items()) 
#??????????????????

img = Image.open("../input/animal-faces/afhq/val/dog/flickr_dog_000045.jpg")

# ?????????resize???224x224??????

img = img.resize((im_width, im_height))
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
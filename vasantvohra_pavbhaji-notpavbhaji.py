# To get the input path.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        break

print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random,os,glob

import numpy as np

import time

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.random import set_random_seed

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Conv2D,Dense, Dropout,MaxPooling2D,GlobalAveragePooling2D

from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam

from tensorflow.keras.regularizers import l1, l2, L1L2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#set_random_seed(0)

#np.random.seed(0)
# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
path = "../input/pavbhaji"
train_datagen = ImageDataGenerator(

        rescale = 1./255,

        rotation_range = 20,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True,

        vertical_flip = True,

        fill_mode='nearest')

validation_datagen = ImageDataGenerator(

        rescale = 1./255)

test_datagen = ImageDataGenerator(

        rescale = 1./255)
img_shape = (224, 224, 3) # default values



train_batch_size = 77 #64

val_batch_size = 33 #32



train_generator = train_datagen.flow_from_directory(

            directory = path + '/train', 

            target_size = (img_shape[0], img_shape[1]),

            batch_size = train_batch_size,

            class_mode = 'categorical',

            color_mode="rgb",

            shuffle = True,

            seed=42) #binary - not working



validation_generator = validation_datagen.flow_from_directory(

            directory = path + '/valid',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            color_mode="rgb",

            shuffle = True) #False



test_generator = test_datagen.flow_from_directory(

            directory = path + '/test',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = 1,

            class_mode = None,

            color_mode="rgb",

            shuffle = False)
for image_batch, label_batch in train_generator:

  break

image_batch.shape, label_batch.shape
print ("Train_generator",train_generator.class_indices)

print ("Validation_generator",validation_generator.class_indices)

print ("Test_generator",test_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))



with open('labels.txt', 'w') as f:

  f.write(labels)

labels = dict((v,k) for k,v in train_generator.class_indices.items())

print("Our Labels",labels)
def Visualize(image,label):

     fig, m_axs = plt.subplots(2, 4, figsize = (16, 8))

     for (img, classs, c_ax) in zip(image, label, m_axs.flatten()):

         img = np.squeeze(img)

         c_ax.imshow(img)

         c_ax.set_title('%s' % labels[np.argmax(classs)])

         c_ax.axis('off')
image,label = next(train_generator)

Visualize(image,label)
image,label = next(validation_generator)

Visualize(image,label)
#image,label = next(test_generator)

#Visualize(image,label)
from tensorflow.keras.applications import InceptionV3

#from tensorflow.keras.applications import VGG16

#from tensorflow.keras.applications import ResNet50

inception = InceptionV3(weights = 'imagenet',

              include_top = False,

              input_shape = img_shape)
print("Number of layers in the inception model: ", len(inception.layers))

# Freeze the layers except the last 30 layers

for layer in inception.layers[:-30]:

    layer.trainable = False
with strategy.scope():

    

# Create the model

    model = Sequential()



     # Add the convolutional base model

    model.add(inception)

    model.add(Conv2D(128, 3, activation='relu'))# kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)

    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())

    # Add new layers

    model.add(Flatten())



    model.add(Dense(512, activation='relu'))

    #model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    # last layer

    model.add(Dense(2, activation='softmax', kernel_regularizer=l2(0.001))) #relu,sigmoid #not 1 -categorical
model.summary()
model.compile(loss='categorical_crossentropy', #binary, Nadam acc doesn't change

              optimizer=Adam(lr=1e-4),

              metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

mc = ModelCheckpoint('inception.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True) #Inception.h5



steps_per_epoch = train_generator.samples//train_generator.batch_size

validation_steps = validation_generator.samples//validation_generator.batch_size

start = time.time()

history = model.fit_generator(

    train_generator,

    steps_per_epoch=steps_per_epoch ,

    epochs=50,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    verbose=1,

    workers=4,

    callbacks=[es, mc])

end = time.time()

print('Execution time: ', end-start)
train_acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

train_loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(train_acc) + 1)



plt.plot(epochs, train_acc, 'b*-', label = 'Training accuracy')

plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, train_loss, 'b*-', label = 'Training loss')

plt.plot(epochs, val_loss, 'r', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
def load_img(img_path):

    from keras.preprocessing import image

    img = image.load_img(img_path, target_size=(224, 224))

    img = image.img_to_array(img, dtype=np.uint8)

    img = np.array(img)/255.0

    

    plt.title("Loaded Image")

    plt.axis('off')

    plt.imshow(img.squeeze())

    return(img)
def prediction(img):

    

     model = tf.keras.models.load_model("inception.h5")

     p = model.predict(img[np.newaxis, ...])

     classes=[]

     prob=[]

     print("\n-------------------Individual Probability--------------------------------\n")

     for i,j in enumerate (p[0],0):

         print(labels[i].upper(),':',round(j*100,2),'%')

         classes.append(labels[i])

         prob.append(round(j*100,2))

         

     def plot_bar_x():

         # this is for plotting purpose

         index = np.arange(len(classes))

         plt.bar(index, prob)

         plt.xlabel('Labels', fontsize=12)

         plt.ylabel('Probability', fontsize=12)

         plt.xticks(index, classes, fontsize=12, rotation=20)

         plt.title('Probability for loaded image')

         plt.show()

     plot_bar_x()
img = load_img(path +'/train/Not_PavBhaji/38618427_2227520140864849_2036571121217699840_n.jpg')
prediction(img)
img = load_img(path +'/train/PavBhaji/20181115_125235.jpg')
prediction(img)
test_steps = test_generator.samples//test_generator.batch_size

test_generator.reset()

model = tf.keras.models.load_model("inception.h5")

prediction = model.predict_generator(test_generator,

                                steps = test_steps,

                                verbose=1)

#import pdb

#pdb.set_trace()

#print(prediction)

pred_binary = [np.argmax(value) for value in prediction] 

pred_binary = np.array(pred_binary)

#pred_binary.reshape(24,1)

#print(pred_binary)



import collections

print(collections.Counter(pred_binary))



##Id = test_generator.index_array

##Id = os.listdir("%s/test/PavBhaji"%path)

##Id.extend(os.listdir("%s/test/Not_PavBhaji"%path))

Id = test_generator.filenames

pred_list_new = [labels[f] for f in pred_binary]

##print(pred_list_new)

##print(len(Id))



test_df = pd.DataFrame({'Image_name': Id,'Predicted_Label': pred_list_new})

test_df.to_csv('submission.csv', header=True, index=False)

test_df
validation_steps = validation_generator.samples//validation_generator.batch_size

model.evaluate_generator(validation_generator,

                        steps = test_steps)
images, label = next(validation_generator)

model = tf.keras.models.load_model("inception.h5")

probabilities = model.predict(images)

Visualize(images,probabilities)
from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns

LABELS=['Not_PavBhaji','PavBhaji']

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(

        confusion_matrix, index=class_names, columns=class_names, 

    )

    fig = plt.figure(figsize=figsize, dpi = 300)

    try:

        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")

    except ValueError:

        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return fig



print_confusion_matrix(confusion_matrix(np.argmax(label,-1),np.argmax(probabilities,-1), labels = range(label.shape[1])), 

                            class_names = LABELS, figsize = (10, 1)).savefig('confusion_matrix.png')



print(classification_report(np.argmax(label,-1), 

                            np.argmax(probabilities,-1), 

                            target_names = LABELS))
from IPython.display import FileLinks

FileLinks('.')
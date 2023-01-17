!rm ./*.hdf5
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

import seaborn as sns

%matplotlib inline

from sklearn.metrics import confusion_matrix



import cv2

import os

import glob
# Input data files are available in the "../input/" directory.

INPUT_PATH = "../input/pneumonia-detection/chest_xray"



# List the files in the input directory.

print(os.listdir(INPUT_PATH))
base_dir = INPUT_PATH

train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')

test_dir = os.path.join(base_dir, 'test')



train_0_dir = os.path.join(train_dir, 'Normal'.upper())

train_1_dir = os.path.join(train_dir, 'Pneumonia'.upper())



val_0_dir = os.path.join(val_dir, 'Normal'.upper())

val_1_dir = os.path.join(val_dir, 'Pneumonia'.upper())



test_0_dir = os.path.join(test_dir, 'Normal'.upper())

test_1_dir = os.path.join(test_dir, 'Pneumonia'.upper())



def get_data_list():

    train_0_list = [os.path.join(train_0_dir, fn) for fn in os.listdir(train_0_dir)]

    train_1_list = [os.path.join(train_1_dir, fn) for fn in os.listdir(train_1_dir)]

    val_0_list = [os.path.join(val_0_dir, fn) for fn in os.listdir(val_0_dir)]

    val_1_list = [os.path.join(val_1_dir, fn) for fn in os.listdir(val_1_dir)]

    test_0_list = [os.path.join(test_0_dir, fn) for fn in os.listdir(test_0_dir)]

    test_1_list = [os.path.join(test_1_dir, fn) for fn in os.listdir(test_1_dir)]



    # list dir numbers

    print('total picture numbers in train_0_dir: ', len(train_0_list))

    print('total picture numbers in train_1_dir: ', len(train_1_list))

    print('total picture numbers in val_0_dir: ', len(val_0_list))

    print('total picture numbers in val_1_dir: ', len(val_1_list))

    print('total picture numbers in test_0_dir: ', len(test_0_list))

    print('total picture numbers in test_1_dir: ', len(test_1_list))



    return (train_0_list, train_1_list, val_0_list, val_1_list, test_0_list, test_1_list)
(train_0_list, train_1_list, val_0_list, val_1_list, test_0_list, test_1_list) = get_data_list()
import random 

(mv_cnt_0, mv_cnt_1) = (300, 300)



if len(val_0_list) < mv_cnt_0:

    mv_list_0 = random.sample(train_0_list, mv_cnt_0)

    mv_list_1 = random.sample(train_1_list, mv_cnt_1)

    train_0_list = [fn for fn in train_0_list if not fn in mv_list_0]

    train_1_list = [fn for fn in train_1_list if not fn in mv_list_1]

    val_0_list += mv_list_0

    val_1_list += mv_list_1

    

    print('total picture numbers in train_0_dir: ', len(train_0_list))

    print('total picture numbers in train_1_dir: ', len(train_1_list))

    print('total picture numbers in val_0_dir: ', len(val_0_list))

    print('total picture numbers in val_1_dir: ', len(val_1_list))

    print('total picture numbers in test_0_dir: ', len(test_0_list))

    print('total picture numbers in test_1_dir: ', len(test_1_list))
(left, top) = (15, 40)

(y1, y2, x1, x2) = (top,top+200, left,left+200)

def image_resize(img_path):

    # print(dataset.shape)

    

    im = cv2.imread(img_path)

    im = cv2.resize(im, (224,224))

    if im.shape[2] == 1:

        # np.dstack(): Stack arrays in sequence depth-wise (along third axis).

        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html

        im = np.dstack([im, im, im])

        

        # ----------------------------------------------------------------------------------------

        # cv2.cvtColor(): The function converts an input image from one color space to another. 

        # [Ref.1]: "cvtColor - OpenCV Documentation"

        #     - https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

        # [Ref.2]: "Python计算机视觉编程- 第十章 OpenCV" 

        #     - https://yongyuan.name/pcvwithpython/chapter10.html

        # ----------------------------------------------------------------------------------------

    x_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    x_image = x_image[y1:y2, x1:x2]

    x_image = cv2.resize(x_image, (150,150))

    # Normalization

    # x_image = x_image.astype(np.float32)/255.

    return x_image
import matplotlib.pyplot as plt

import matplotlib.image as mimg

%matplotlib inline

import cv2

import numpy as np
fn_list_0 = train_0_list[:4]

fn_list_1 = train_1_list[:4]



fig, ax = plt.subplots(2, 4, figsize=(20,10))

for i, axi in enumerate(ax.flat):

    img_path = None

    if i < 4:

        img_path = fn_list_0[i]

    else:

        img_path = fn_list_1[i-4]

    img = image_resize(img_path)#.astype(np.uint8)

    axi.imshow(img, cmap='bone')

    axi.set_title(img_path.split('/')[-1])

    axi.set(xticks=[], yticks=[])
def create_dataset(img_path_list_0, img_path_list_1, return_fn = False):

    # list of the paths of all the image files

    normal = img_path_list_0

    pneumonia = img_path_list_1



    # --------------------------------------------------------------

    # Data-paths' format in (img_path, label) 

    # labels : for [ Normal cases = 0 ] & [ Pneumonia cases = 1 ]

    # --------------------------------------------------------------

    normal_data = [(image, 0) for image in normal]

    pneumonia_data = [(image, 1) for image in pneumonia]



    image_data = normal_data + pneumonia_data



    # Get a pandas dataframe for the data paths 

    image_data = pd.DataFrame(image_data, columns=['image', 'label'])

#     print(image_data.head(5))

    # Shuffle the data 

    image_data = image_data.sample(frac=1., random_state=100).reset_index(drop=True)

    

    # Importing both image & label datasets...

    (x_images, y_labels) = ([image_resize(image_data.iloc[i][0]) for i in range(len(image_data))], 

                         [image_data.iloc[i][1] for i in range(len(image_data))])



    # Convert the list into numpy arrays

    x_images = np.array(x_images)

    y_labels = np.array(y_labels)

    

    print("Total number of images: ", x_images.shape)

    print("Total number of labels: ", y_labels.shape)

    

    if not return_fn:

        return (x_images, y_labels)

    else:

        return (x_images, y_labels, image_data.image.values)
# Import train dataset...

(x_train, y_train) = create_dataset(train_0_list, train_1_list)



print(x_train.shape)

print(y_train.shape)
# Import val dataset...

(x_val, y_val) = create_dataset(val_0_list, val_1_list)
from tensorflow.keras.applications import VGG16

# weights: None 代表随机初始化， 'imagenet' 代表加载在 ImageNet 上预训练的权值

# we could customize input_shape when include_top = False, otherwise input_shape need to be (299, 299, 3); but width and height cannot be less than 71

conv_base = VGG16(weights='../input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(150, 150, 3))

conv_base.summary()
# create data generator (without data augment)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

import tensorflow.keras.backend as K



# rescale all image by 1/255 

data_batch_size = 20



def extract_feature(X_array, y_array, sample_count):

    features_list = np.zeros(shape=(sample_count, 4, 4, 512))

    labels_list = np.zeros(shape=(sample_count))

    datagen = ImageDataGenerator(rescale=1./255)

    datagen.fit(X_array)

    data_generator = datagen.flow(X_array, y_array, batch_size=data_batch_size)

    i = 0

    for data_batch, labels_batch in data_generator:

        feature_map = conv_base.predict(data_batch) # use conv_base to extract feature map

        features_list[i*data_batch_size: (i+1)*data_batch_size] = feature_map

        labels_list[i*data_batch_size: (i+1)*data_batch_size] = labels_batch

        i += 1

        if i*data_batch_size >= sample_count:

            break

    return (features_list, labels_list)



def get_f1(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
(train_features, train_labels) = extract_feature(x_train, y_train, len(y_train))

(valid_features, valid_labels) = extract_feature(x_val, y_val, len(y_val))
print((train_features.shape, train_labels.shape), (valid_features.shape, valid_labels.shape))
train_features_flatten = np.reshape(train_features, (train_features.shape[0], 4*4*512))

valid_features_flatten = np.reshape(valid_features, (valid_features.shape[0], 4*4*512))

train_features_flatten.shape
from sklearn import metrics

from sklearn.metrics import confusion_matrix

import seaborn as sns



def get_pred_score(y_true, y_pred):

    mat = confusion_matrix(y_true, y_pred)

    print(mat)



    plt.figure(figsize=(8,6))

    sns.heatmap(mat, square=False, annot=True, fmt ='d', cbar=True, annot_kws={"size": 16})

    plt.title('0 : Normal   1 : Pneumonia', fontsize = 20)

    plt.xticks(fontsize = 16)

    plt.yticks(fontsize = 16)

    plt.xlabel('predicted value', fontsize = 20)

    plt.ylabel('true value', fontsize = 20)

    plt.show()



    tn, fp, fn, tp = mat.ravel()

    print('\ntn = {}, fp = {}, fn = {}, tp = {} '.format(tn, fp, fn, tp))



    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    f1_score = 2. * precision * recall / (precision + recall)

    f2_score = 5. * precision * recall / (4. * precision + recall)



    print("Test Recall of the model \t = {:.4f}".format(recall))

    print("Test Precision of the model \t = {:.4f}".format(precision))

    print("Test Accuracy of the model \t = {:.4f}".format(accuracy))

    print("Test F1 score of the model \t = {:.4f}".format(f1_score))

    print("Test F2 score of the model \t = {:.4f}".format(f2_score))
from tensorflow.keras import layers, models

from tensorflow.keras import optimizers



use_flatten = True

model = models.Sequential()

if use_flatten:

    model.add(layers.Dense(256, activation='relu', input_shape= (4,4,512)))

    model.add(layers.Flatten())

else:

    model.add(layers.Dense(256, activation='relu', input_dim = 4*4*512))



model.add(layers.Dropout(0.5)) 

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
default_lr = 1e-4 

adp_optimizer = optimizers.RMSprop(lr=default_lr, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
from tensorflow.keras.callbacks import ModelCheckpoint

# Define a checkpoint callback :

checkpoint_name = 'Weights-m1-{epoch:03d}--{val_loss:.5f}.hdf5'

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]
if use_flatten:

    history = model.fit(train_features, train_labels, batch_size=20, epochs=30, validation_data=(valid_features, valid_labels), callbacks=callbacks_list)

else:

    history = model.fit(train_features_flatten, train_labels, batch_size=20, epochs=30, validation_data=(valid_features_flatten, valid_labels), callbacks=callbacks_list)
import matplotlib.pyplot as plt





acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

loss = history.history['loss']

val_loss = history.history['val_loss']

f1 = history.history['get_f1']

val_f1 = history.history['val_get_f1']



plt.plot(epochs, acc, 'bo', label='Train Acc')

plt.plot(epochs, val_acc, 'b', label='Validation Acc')

plt.title('Accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, f1, 'bo', label='Train F1')

plt.plot(epochs, val_f1, 'b', label='Validation F1')

plt.title('F1 score')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Train Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Loss')

plt.legend()

plt.figure()



plt.show()
# Import train dataset...

(x_test, y_test, test_fns) = create_dataset(test_0_list, test_1_list, return_fn=True)
(test_features, test_labels) = extract_feature(x_test, y_test, len(y_test))

print((train_features.shape, train_labels.shape))

test_features_flatten = np.reshape(test_features, (test_features.shape[0], 4*4*512))
# Load best weight of model

from pathlib import Path

w_fnl = [str(fn) for fn in Path('./').glob('Weights-m1-*.hdf5')]

w_fnl.sort(reverse=True)

wights_file = w_fnl[0] # choose the best checkpoint 

model.load_weights(wights_file) # load it

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
if use_flatten:

    pred_prob = model.predict(test_features, batch_size=data_batch_size)

else:

    pred_prob = model.predict(test_features_flatten, batch_size=data_batch_size)

pred_res = np.asarray([1 if x > 0.5 else 0 for x in [x[0] for x in pred_prob]]) 

get_pred_score(test_labels, pred_res)
# conv_base = VGG16(weights='../input/vgg16-weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(150, 150, 3))

model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten()) # 先Flatten loss稍好，且可以減少sigmoid層的param數

model.add(layers.Dense(256, activation='relu'))

# model.add(layers.Flatten())

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
# Freezing a layer or set of layers means preventing their weights from being updated during training.

print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

conv_base.trainable = False

print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
# show trainable weights

[x.name for x in model.trainable_weights]
# use ImageGenerator to generate more training data

train_datagen = ImageDataGenerator(

    rescale=1./255,  # Rescales all images by 1/255

    rotation_range = 10,

    width_shift_range = 0.2, height_shift_range = 0.2,

    fill_mode = 'nearest', shear_range = 0.2,

    zoom_range = 0.2, horizontal_flip=False, 

)

train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(rescale=1./255) #validation set no need to augment

val_datagen.fit(x_val)



train_generator = train_datagen.flow(x_train, y_train, batch_size=32) #increase batch size to 32

val_generator = val_datagen.flow(x_val, y_val, batch_size=32) #increase batch size to 32
model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
# Define a checkpoint callback for method2:

checkpoint_name = 'Weights-m2-{epoch:03d}--{val_loss:.5f}.hdf5'

checkpoint2 = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list2 = [checkpoint2]
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=val_generator, validation_steps=20, callbacks=callbacks_list2)
import matplotlib.pyplot as plt





acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

loss = history.history['loss']

val_loss = history.history['val_loss']

f1 = history.history['get_f1']

val_f1 = history.history['val_get_f1']



plt.plot(epochs, acc, 'bo', label='Train Acc')

plt.plot(epochs, val_acc, 'b', label='Validation Acc')

plt.title('Accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, f1, 'bo', label='Train F1')

plt.plot(epochs, val_f1, 'b', label='Validation F1')

plt.title('F1 score')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Train Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Loss')

plt.legend()

plt.figure()



plt.show()
test_data = []

test_labels = []

for (test_img, label) in zip(x_test, y_test):

    test_data.append(test_img.astype(np.float32)/255)

    test_labels.append(label)



test_data = np.array(test_data)

test_labels = np.array(test_labels)



print("Total number of test examples: ", test_data.shape)

print("Total number of labels:", test_labels.shape)
def predict(model, test_data):

    pred_prob = model.predict(test_data, batch_size=data_batch_size)

    pred_res = np.asarray([1 if x > 0.5 else 0 for x in [x[0] for x in pred_prob]]) 

    return pred_res
# Load best weight of model



w_fnl = [str(fn) for fn in Path('./').glob('Weights-m2-*.hdf5')]

w_fnl.sort(reverse=True)

wights_file = w_fnl[0] # choose the best checkpoint 

model.load_weights(wights_file) # load it

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
get_pred_score(test_labels, predict(model, test_data))
conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:

    if layer.name == 'block5_conv1':

#     if layer.name == 'block4_conv1':

        set_trainable = True

      # set trainable = True for layers after block5_conv1

    if set_trainable:

        layer.trainable = True

        print(layer.name)

    else:

        layer.trainable = False
model.summary()
# use very low learning rate

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
# Define a checkpoint callback for method3:

checkpoint_name = 'Weights-m3-{epoch:03d}--{val_loss:.5f}.hdf5'

checkpoint3 = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list3 = [checkpoint3]
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=val_generator, validation_steps=20, callbacks=callbacks_list3)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = range(1, len(acc)+1)

loss = history.history['loss']

val_loss = history.history['val_loss']

f1 = history.history['get_f1']

val_f1 = history.history['val_get_f1']



plt.plot(epochs, acc, 'bo', label='Train Acc')

plt.plot(epochs, val_acc, 'b', label='Validation Acc')

plt.title('Accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, f1, 'bo', label='Train F1')

plt.plot(epochs, val_f1, 'b', label='Validation F1')

plt.title('F1 score')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Train Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Loss')

plt.legend()

plt.figure()



plt.show()
# Load best weight of model



w_fnl = [str(fn) for fn in Path('./').glob('Weights-m3-*.hdf5')]

w_fnl.sort(reverse=True)

wights_file = w_fnl[0] # choose the best checkpoint 

print('apply weight files: ', wights_file)

model.load_weights(wights_file) # load it

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
y_pred = predict(model, test_data)

get_pred_score(test_labels, y_pred)
result_df = pd.DataFrame({'fn':test_fns, 'label': test_labels, 'pred': y_pred})

false_df = result_df[result_df.pred != result_df.label]

false_df.shape
false_df.head(10)
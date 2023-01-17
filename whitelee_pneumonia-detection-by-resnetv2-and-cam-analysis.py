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
from keras.applications.resnet50 import ResNet50

from keras.applications.vgg16 import VGG16

from keras.applications.inception_resnet_v2 import InceptionResNetV2

# base_model = ResNet50(weights='../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(150, 150, 3), include_top = False, pooling = 'avg')

# base_model = VGG16(weights='../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(150, 150, 3), include_top = False)

base_model = InceptionResNetV2(weights='../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5', 

                   input_shape=(150, 150, 3), include_top = False)

base_model.summary()
# create data generator (without data augment)

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

import keras.backend as K



# rescale all image by 1/255 

data_batch_size = 20



def get_f1(y_true, y_pred): #taken from old keras source code

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    recall = true_positives / (possible_positives + K.epsilon())

    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val
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
from keras import layers, models, Model

from keras import optimizers

from keras.callbacks import ModelCheckpoint, EarlyStopping
# Import train dataset...

(x_test, y_test, test_fns) = create_dataset(test_0_list, test_1_list, return_fn=True)
x = base_model.output

# x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)

x = layers.Dense(512, activation = 'relu')(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = Model(base_model.input,x)
# # Freezing a layer or set of layers means preventing their weights from being updated during training.

# freezing_layer = None

# model.trainable = False

print('This is the number of trainable weights: ', len(model.trainable_weights))

# for layer in reversed(model.layers):

#     # check to see if the layer has a 4D output

#     if len(layer.output_shape) == 4:

#         freezing_layer = layer.name

#         break

# print('Freezing layer = ', freezing_layer)

# set_trainable = False

# for layer in model.layers:

#     if layer.name == freezing_layer:

#         set_trainable = True

#       # set trainable = True for layers after block5_conv1

#     if set_trainable:

#         layer.trainable = True

#         print(layer.name)

#     else:

#         layer.trainable = False

# print('This is the number of trainable weights after freezing the conv base:', len(model.trainable_weights))
model.summary()
# show trainable weights

# [x.name for x in model.trainable_weights]
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
default_lr = 1e-4 

adp_optimizer = optimizers.RMSprop(lr=default_lr, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
# Define a checkpoint callback for method2:

checkpoint_name = 'Weights-m2-{epoch:03d}--{val_loss:.5f}.hdf5'

checkpoint2 = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

es = EarlyStopping(monitor='val_loss', patience=5)

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

    return (pred_res, [x[0] for x in pred_prob])
# Load best weight of model

from pathlib import Path

w_fnl = [str(fn) for fn in Path('./').glob('Weights-m2-*.hdf5')]

w_fnl.sort(reverse=True)

wights_file = w_fnl[0] # choose the best checkpoint 

model.load_weights(wights_file) # load it

model.compile(optimizer=adp_optimizer, loss="binary_crossentropy", metrics=["accuracy", get_f1])
(y_pred, y_pred_prob) = predict(model, test_data)

get_pred_score(test_labels, y_pred)
pred_result = pd.DataFrame({'imgPath': test_fns, 'label': test_labels, 'pred': y_pred, 'pred_prob': y_pred_prob})

pred_result['fn'] = pred_result.imgPath.apply(lambda ip: ip.split('/')[-1])

pred_result.head()
false_result = pred_result[pred_result.pred != pred_result.label]

true_0_fns = pred_result[(pred_result.pred == pred_result.label) & (pred_result.label == 0)].fn.values

true_1_fns = pred_result[(pred_result.pred == pred_result.label) & (pred_result.label == 1)].fn.values

false_0_fns = false_result[false_result.label == 0].fn.values

false_1_fns = false_result[false_result.label == 1].fn.values

print(true_0_fns.shape, true_1_fns.shape, false_0_fns.shape, false_1_fns.shape)
# image_path = os.path.join(test_0_dir, false_0_fns[0])

fn = false_0_fns[1]

folder = test_0_dir

image_path = os.path.join(folder, fn)

print(image_path)

# img = image.load_img(image_path, target_size=(224, 224))

img = image_resize(image_path)

img = img.astype(np.float32)/255

img_x = np.expand_dims(img, axis=0)

img_x.shape
from keras.models import Model

import tensorflow as tf

import keras.backend as K



class GradCAM:

    def __init__(self, model, classIdx=0, layerName=None):

        # store the model, the class index used to measure the class

        # activation map, and the layer to be used when visualizing

        # the class activation map

        self.model = model

        self.classIdx = classIdx

        self.layerName = layerName

        self.sess = tf.compat.v1.Session()

        # if the layer name is None, attempt to automatically find

        # the target output layer

        if self.layerName is None:

            self.layerName = self.find_target_layer()



    def find_target_layer(self):

        # 尋找最後一層Conv layer

        # attempt to find the final convolutional layer in the network

        # by looping over the layers of the network in reverse order

        for layer in reversed(self.model.layers):

            # check to see if the layer has a 4D output

            if len(layer.output_shape) == 4:

                # model中出現的最後一個輸出維度為4的層即尋找目標

                return layer.name

        # otherwise, we could not find a 4D layer so the GradCAM

        # algorithm cannot be applied

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")



    def compute_heatmap(self, image, eps=1e-8):

        # construct our gradient model by supplying (1) the inputs

        # to our pre-trained model, (2) the output of the (presumably)

        # final 4D layer in the network, and (3) the output of the

        # softmax activations from the model

#         gradModel = Model(

#             inputs=[self.model.inputs],

#             outputs=[self.model.get_layer(self.layerName).output,

#                 self.model.output])

            # record operations for automatic differentiation

        

        pred = self.model.predict(image)

        predictions = self.model.output

        # model 對輸入作完預測後取出要繪制heatmap的類別，因我們的model為二元分類，故classIdx固定為0

        loss = predictions[:, self.classIdx]

        convOutputs = self.model.get_layer(self.layerName).output



        # 用 gradients 函式計算梯度值作為後面畫heatmap的權重

        # use automatic differentiation to compute the gradients

        grads = K.gradients(loss, convOutputs)[0]



        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        iterate = K.function([self.model.input],[pooled_grads, convOutputs[0]])

        (pooled_grads_value, conv_layer_output_value) = iterate([image])

        for i in range(512):

            # 對conv layer的輸出乘上權重以作為繪制heatmap的raw data

            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        

        heatmap = np.mean(conv_layer_output_value, axis=-1)

        # grab the spatial dimensions of the input image and resize

        # the output class activation map to match the input image

        # dimensions

        (w, h) = (image.shape[2], image.shape[1])

        # heatmap = cv2.resize(cam.numpy(), (w, h))

        heatmap = cv2.resize(heatmap, (w, h))

        # 正規化 heatmap的raw data使值落在0~255之間(image data的合理範圍)

        # normalize the heatmap such that all values lie in the range

        # [0, 1], scale the resulting values to the range [0, 255],

        # and then convert to an unsigned 8-bit integer

        numer = heatmap - np.min(heatmap)

        denom = (heatmap.max() - heatmap.min()) + eps

        heatmap = numer / denom

        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function

        return (heatmap, pred)



    def overlay_heatmap(self, heatmap, image, alpha=0.5,

        colormap=cv2.COLORMAP_VIRIDIS):

        # apply the supplied color map to the heatmap and then

        # overlay the heatmap on the input image

        heatmap = cv2.applyColorMap(heatmap, colormap)

        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        # return a 2-tuple of the color mapped heatmap and the output,

        # overlaid image

        return (heatmap, output)
cam = GradCAM(model)

(heatmap, _) = cam.compute_heatmap(img_x)
img = image_resize(image_path)

print(heatmap.shape, type(heatmap))

print(img[:, :, 0].shape, type(img[:, :, 0]))
(heatmap, output) = cam.overlay_heatmap(np.dstack([heatmap, heatmap, heatmap]), img, alpha=0.2)
plt.imshow(heatmap) # show heatmap
plt.imshow(output) # 疊合圖
# Create global CAM class

cam = GradCAM(model)
def get_heatmap_with_pic(base_dir, fn, cam_model=cam):

#     print(fn)

    image_path = os.path.join(base_dir, fn)

    title1 = fn

    img = image_resize(image_path)

    img_nz = img.astype(np.float32)/255

    img_x = np.expand_dims(img_nz, axis=0)

    (heatmap, pred_prob) = cam_model.compute_heatmap(img_x)

    (heatmap, output) = cam_model.overlay_heatmap(np.dstack([heatmap, heatmap, heatmap]), img, alpha=0.2)

    title2 = 'pred_prob: {}'.format(pred_prob)

    title3 = 'combined'

    return (img, title1, heatmap, title2, output, title3)
# 使用 subplots 將原始圖、heatmap及疊合圖並排呈現

def draw_heatmap_on_plt(base_dir, fn_list, cat='tp', cam_model=cam):

    # Draw heatmap for true cases

    pics_per_row = 2

    cols = 3 * pics_per_row

    rows = int(len(fn_list) / pics_per_row)

    fig, ax = plt.subplots(rows, cols, figsize=(cols*5,rows*5))

    for i, axi in enumerate(ax.flat):

        title = ''

        show_img = None

        idx = i % 3

        if idx == 0:

            (img, title1, heatmap, title2, superimposed_img, title3) = get_heatmap_with_pic(base_dir, fn_list[int(i / 3)])

            show_img = img

            title = '{} ({})'.format(title1, cat)

        elif idx == 1:

            show_img = heatmap

            title = title2

        else:

            show_img = superimposed_img

            title = title3

        axi.imshow(show_img, cmap='bone')

        axi.set_title(title)

        axi.set(xticks=[], yticks=[])
import datetime

disp_num = 15 #每個類別取15張原圖

pic_sec = 20 # 畫一張 heatmap 約要20秒(以kaggle的notebook規格)
disp_cnt = min(disp_num, len(true_0_fns))

print('Will take around {} secs from {}'.format(disp_cnt*pic_sec, str(datetime.datetime.now())))

draw_heatmap_on_plt(test_0_dir, true_0_fns[:disp_cnt], 'tn')
disp_cnt = min(disp_num, len(true_1_fns))

print('Will take around {} secs from {}'.format(disp_cnt*pic_sec, str(datetime.datetime.now())))

draw_heatmap_on_plt(test_1_dir, true_1_fns[:disp_cnt], 'tp')
disp_cnt = min(disp_num, len(false_0_fns))

print('Will take around {} secs from {}'.format(disp_cnt*pic_sec, str(datetime.datetime.now())))

draw_heatmap_on_plt(test_0_dir, false_0_fns[:disp_cnt], 'fn')
disp_cnt = min(disp_num, len(false_1_fns))

print('Will take around {} secs from {}'.format(disp_cnt*pic_sec, str(datetime.datetime.now())))

draw_heatmap_on_plt(test_1_dir, false_1_fns[:disp_cnt], 'fp')
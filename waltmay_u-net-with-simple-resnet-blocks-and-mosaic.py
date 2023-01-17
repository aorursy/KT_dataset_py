import os
import sys
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")

%matplotlib inline

# import cv2
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
t_start = time.time()
#print(os.listdir("../input/trained-unet-model/"))
version = 5
basic_name = f'Unet_resnet_v{version}'
save_model_name = basic_name + '.model'
previous_model_name = "../input/u-net-with-simple-resnet-blocks-v2-new-loss/" + save_model_name
submission_file = basic_name + '.csv'
data_source = "../input/tgs-salt-identification-challenge/"
stored_trained_model = "../input/trained-unet-model/Unet_resnet_v5.model"
fMosaic = "../input/trained-unet-model/mosaic.csv"
fTrain = "../input/trained-unet-model/train-files.csv"
fTest = "../input/trained-unet-model/test-files.csv"


#print(save_model_name)
#print(previous_model_name)
#print(submission_file)
img_size_ori = 101
img_size_target = 101

def upsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):# not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
mosaic_df=pd.read_csv(fMosaic)
train_files_df = pd.read_csv(fTrain)
test_files_df = pd.read_csv(fTest)

train_files_df.insert(0, 'img_id', range(0,len(train_files_df)))
test_files_df.insert(0, 'img_id', range(len(train_files_df),len(train_files_df)+len(test_files_df)))
all_files_df = pd.concat([train_files_df,test_files_df],sort=False)

all_files_df[['img_name','a','b']]=all_files_df['x'].str.partition(".")



#all_files_df.head(4010)
# Loading of training/testing ids and depths
train_df = pd.read_csv(data_source + "train.csv", index_col="id", usecols=[0])
depths_df = pd.read_csv(data_source + "depths.csv", index_col="id")
train_df = train_df.join(depths_df)

#test_df = pd.read_csv(data_source + "test", index_col="id", usecols=[0])
test_df = depths_df[~depths_df.index.isin(train_df.index)]

#train_df.head()
test_df["images"] = [np.array(load_img(data_source + "test/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(test_df.index)]

train_df["images"] = [np.array(load_img(data_source + "train/images/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
train_df["masks"] = [np.array(load_img(data_source + "train/masks/{}.png".format(idx), grayscale=True)) / 255 for idx in tqdm_notebook(train_df.index)]
#train_df.head()
#train_df[['xof','yof','mos_num']]
#train_df = pd.merge(train_df,mosaic_df,how = 'left',left_on='img_id', right_on = 'img_id')
#test_df = pd.merge(test_df,mosaic_df,how = 'left',left_on='img_id', right_on = 'img_id')


#mosaic_df.head()
all_arr = pd.concat([train_df,test_df],sort=False)
all_arr.insert(0, 'orig_id', range(1,len(all_arr)+1))
#all_arr.head(20)
all_arr = pd.merge(all_arr,all_files_df,how = 'left',left_index=True, right_on = 'img_name')
#all_arr.head()
#all_arr = pd.merge(all_arr,mosaic_df,how = 'left',left_on='img_id', right_on = 'img_id')
#all_arr.head(4010)
train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i
        
train_df["coverage_class"] = train_df.coverage.map(cov_to_class)

#test = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)
# Create train/validation split stratified by salt coverage
ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
    train_df.index.values,
    np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
    train_df.coverage.values,
    train_df.z.values,
    test_size=0.2, stratify=train_df.coverage_class, random_state= None)
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

# Build model
def build_model(input_layer, start_neurons, DropoutRatio = 0.5):
    # 101 -> 50
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1,start_neurons * 1)
    conv1 = residual_block(conv1,start_neurons * 1, True)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2,start_neurons * 2)
    conv2 = residual_block(conv2,start_neurons * 2, True)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3,start_neurons * 4)
    conv3 = residual_block(conv3,start_neurons * 4, True)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = residual_block(conv4,start_neurons * 8, True)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm,start_neurons * 16)
    convm = residual_block(convm,start_neurons * 16, True)
    
    # 6 -> 12
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = residual_block(uconv4,start_neurons * 8, True)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = residual_block(uconv3,start_neurons * 4, True)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = residual_block(uconv2,start_neurons * 2, True)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = residual_block(uconv1,start_neurons * 1, True)
    
    #uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = Conv2D(1, (1,1), padding="same", activation=None)(uconv1)
    output_layer =  Activation('sigmoid')(output_layer_noActi)
    
    return output_layer
def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch]>0, B[batch]>0
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
#             metric.append(0)
#             continue
#         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
#             metric.append(1)
#             continue
        
        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10 )/ (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred>0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred >0], tf.float64)
# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss
#Data augmentation
x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
print(x_train.shape)
print(y_valid.shape)
# This will load a stored trained model or the last trained model
from pathlib import Path

if Path(previous_model_name).is_file():
    print("Using previous sucessful run's model")
    model2 = load_model(previous_model_name,custom_objects={'my_iou_metric_2': my_iou_metric,
                                                        'lovasz_loss': lovasz_loss,
                                                        'my_iou_metric': my_iou_metric})
else:
    print("Using stored trained model")
    model2 = load_model(stored_trained_model,custom_objects={'my_iou_metric_2': my_iou_metric,
                                                        'lovasz_loss': lovasz_loss,
                                                        'my_iou_metric': my_iou_metric})
# Use this one if you want to start from a pre-trained model

# remove layter activation layer and use losvasz loss
input_x = model2.layers[0].input

output_layer = Activation('sigmoid',name='output_activaton')(model2.layers[-1].output)
model1 = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

#model1.summary()
#early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

epochs = 10
batch_size = 32
history = model1.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr], 
                    verbose=2)
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_loss.grid(True,axis = 'y')
ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")
ax_score.legend()
ax_score.grid(True,axis = 'y')
model1 = load_model(save_model_name,custom_objects={'my_iou_metric': my_iou_metric})
# remove layter activation layer and use losvasz loss
input_x = model1.layers[0].input

output_layer = model1.layers[-1].input
model = Model(input_x, output_layer)
c = optimizers.adam(lr = 0.01)

# lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
# Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

#model.summary()
early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=30, verbose=1)
model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
epochs = 50
batch_size = 32

history = model.fit(x_train, y_train,
                    validation_data=[x_valid, y_valid], 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                    verbose=2)
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_loss.legend()
ax_loss.grid(True,axis = 'y')
ax_score.plot(history.epoch, history.history["my_iou_metric_2"], label="Train score")
ax_score.plot(history.epoch, history.history["val_my_iou_metric_2"], label="Validation score")
ax_score.legend()
ax_score.grid(True,axis = 'y')
model = model2
model = load_model(save_model_name,custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                   'lovasz_loss': lovasz_loss})


def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
    x_test_reflect =  np.array([np.fliplr(x) for x in x_test])
    preds_test = model.predict(x_test).reshape(-1, img_size_target, img_size_target)
    preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target, img_size_target)
    preds_test += np.array([ np.fliplr(x) for x in preds_test2_refect] )
    return preds_test/2
preds_train = predict_result(model,x_train,img_size_target)
preds_valid = predict_result(model,x_valid,img_size_target)
#Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in


    true_objects = 2
    pred_objects = 2

    #  if all zeros, original code  generate wrong  bins [-0.5 0 0.5],
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0,0.5,1], [0,0.5, 1]))
#     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    #print(temp1)
    intersection = temp1[0]
    #print("temp2 = ",temp1[1])
    #print(intersection.shape)
   # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    #print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels,bins=[0,0.5,1])[0]
    #print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0,0.5,1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    intersection[intersection == 0] = 1e-9
    
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


## Scoring for last model, choose threshold by validation data 
thresholds_ori = np.linspace(0.3, 0.7, 31)
# Reverse sigmoid function: Use code below because the  sigmoid activation was removed
thresholds = np.log(thresholds_ori/(1-thresholds_ori)) 

# ious = np.array([get_iou_vector(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
# print(ious)
ious = np.array([iou_metric_batch(y_valid, preds_valid > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)
# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0)
    if u == 0:
        return 1.0
    return i/u

def plot_sample(X, y, preds):
    ix = random.randint(0, len(X))
    
    iou_score = iou(y[ix,:,:,0],preds[ix]>threshold_best)
    
    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Salt Pred. IOU: ' + str(round(iou_score,5)));
    
    i = ((preds[ix]>threshold_best)*y[ix,:,:,0]) >0
    u = ((preds[ix]>threshold_best) + y[ix,:,:,0]) >0
    
    #print(i)
    #print(preds[ix] + y[ix,:,:,0])
    #print(preds[ix,0,0:5]-preds[ix,0,1:6])
    
    umi = u!=i
    
    #print(y[ix,:,:,0].shape)
    
     
    ax[3].matshow(u, vmin=0, vmax=1, alpha = 1, cmap = 'Reds')
    ax[3].matshow(i, vmin=0, vmax=1, cmap = 'Greens', alpha = 0.5)
    ax[3].set_title('Intercection and Union')
"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
x_test = np.array([(np.array(load_img(data_source + "/test/images/{}.png".format(idx), grayscale = True))) / 255 for idx in tqdm_notebook(test_df.index)]).reshape(-1, img_size_target, img_size_target, 1)


x_all_train = np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

preds_train = predict_result(model,x_all_train,img_size_target)

preds_test = predict_result(model,x_test,img_size_target)
#all_arr.head()
all_preds = list(np.concatenate((preds_train,preds_test)))
#print(all_arr.shape)
preds_df = pd.DataFrame(pd.Series(all_preds), columns = ['pred_masks'])
preds_df.insert(0, 'orig_id', range(1,len(preds_df)+1))

all_arr = pd.merge(all_arr,preds_df,how = 'left',left_on = 'orig_id', right_on = 'orig_id')
#all_arr['pred_masks']=pd.Series(all_preds)
#all_arr.head()#4010)
#mosaic_df.head()
all_arr_saved = all_arr.copy()
#all_arr = all_arr_saved.copy()
def plot_mosaic(index,visualize = 1, train = 1):
    #fig, ax = plt.plot()
    mos = mosaic_df.loc[mosaic_df['mid']==index]

    max_x = int(mos['x_offset'].max())+1
    max_y = int(mos['y_offset'].max())+1
    
    max_x = max(max_x,max_y)
    max_y = max_x
    
    if visualize: fig, ax = plt.subplots(max_y, max_x, figsize=(20, 20))
    #print(max_x,max_y)

    #print(mos.head(10))

    #for index, row in df.iterrows():
    #   print row['c1'], row['c2']
    #print(test_df[all_arr['img_id']==0]['images'])
    image = all_arr.loc[all_arr['img_id']==(1)]['images'].iloc[0]*0
    for i in range(max_x):
        for j in range(max_y):
            #ax[j,i].imshow(image, cmap='seismic')
            if visualize: ax[j,i].axis('off')
            #Here's where I adjust images based on surroundings
            
            img_center = mos.loc[mos['y_offset']==j]
            img_center = img_center.loc[img_center['x_offset']==i] 
            
            if img_center.shape[0] > 0:
                           
                target_img_id = (img_center['img_id'].iloc[0]-1)
                top_pred_mask_center = all_arr.loc[all_arr['img_id']==target_img_id]['pred_masks'].iloc[0]
                np_center_mask = np.array(top_pred_mask_center)
            
            if i>0:
                img_left = mos.loc[mos['y_offset']==j]
                img_left = img_left.loc[img_left['x_offset']==(i-1)]
                if img_left.shape[0] > 0 and img_center.shape[0] > 0:
                    left_img_id = (img_left['img_id'].iloc[0]-1)
                    pred_mask_left = all_arr.loc[all_arr['img_id']==left_img_id]['pred_masks'].iloc[0]
                    
                    np_pred_mask_left = np.array(pred_mask_left)
                    
                    right_column_of_left_mask = np_pred_mask_left[:,-1]
                    
                    left_column_of_center_mask = np_center_mask[:,0]
                    
                    lr_diff = np.mean(left_column_of_center_mask)-np.mean(right_column_of_left_mask)
                    
                    new_left_mask = pred_mask_left - lr_diff*0.5
                    new_right_mask = top_pred_mask_center + lr_diff*0.5
                    
                    if train:
                        all_arr.loc[all_arr['img_id']==target_img_id,'pred_masks'] = [new_right_mask]
                        all_arr.loc[all_arr['img_id']==left_img_id,'pred_masks'] = [new_left_mask]
                    
                    
            if j>0:
                img_above = mos.loc[mos['y_offset']==j-1]
                img_above = img_above.loc[img_above['x_offset']==i]
                
                
                if img_above.shape[0] > 0 and img_center.shape[0] > 0: #This one increase salt threshold if the one below
                    last_pred_mask_above = all_arr.loc[all_arr['img_id']==(img_above['img_id'].iloc[0]-1)]['pred_masks'].iloc[0]
                    last_pred_mask_row_above = last_pred_mask_above[-1]
                    
                    
                    top_pred_mask_row_center = top_pred_mask_center[0]
                    top_bot_mean_diff = np.mean(last_pred_mask_row_above)-np.mean(top_pred_mask_row_center)
                    new_center_mask = top_pred_mask_center + top_bot_mean_diff*1.
                    
                    if train: 
                        all_arr.loc[all_arr['img_id']==target_img_id,'pred_masks'] = [new_center_mask]#pd.Series(list(new_mask))
                    
                    #print('df value: ',all_arr.loc[all_arr['img_id']==target_img_id,'pred_masks'])
                    
    if visualize:
        for index, im in mos.iterrows():
            #print(im)
            x_off = int(im['x_offset'])
            y_off = int(im['y_offset'])
            #print(x_off,y_off)
            #print(im['img_id'])
            #print(all_arr.loc[all_arr['img_id']==1]['images'])#all_arr['img_id']==im['index']])
            image = all_arr.loc[all_arr['img_id']==(im['img_id']-1)]['images'].iloc[0]
            pred_mask = all_arr.loc[all_arr['img_id']==(im['img_id']-1)]['pred_masks'].iloc[0] >threshold_best
            mask = all_arr.loc[all_arr['img_id']==(im['img_id']-1)]['masks'].iloc[0]

            ax[y_off,x_off].imshow(image, cmap='Greys')
            ax[y_off,x_off].imshow(pred_mask, alpha = 0.2, cmap = 'Greens', vmin=0, vmax=1)
            if (im['img_id']<4000):
                ax[y_off,x_off].imshow(mask, alpha = 0.2, cmap = 'seismic')

        #print(mos.shape)
        #mos.head(100)
        #plt.tight_layout()
        plt.subplots_adjust(hspace=0,wspace = 0)
        plt.show()
#plot_mosaic(1)
plot_mosaic(100)
for i in tqdm_notebook(range(1,200)):
    plot_mosaic(i,0,1)

# updates pred_test based on mosiac findings
#print(np.array(all_arr.loc[all_arr['img_id']>3999,'pred_masks'].tolist()).shape)

preds_test = np.array(all_arr.loc[all_arr['img_id']>3999,'pred_masks'].tolist())
preds_train = np.array(all_arr.loc[all_arr['img_id']<=3999,'pred_masks'].tolist())
y_train = np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1)

ious = np.array([iou_metric_batch(y_train, preds_train > threshold) for threshold in tqdm_notebook(thresholds)])
print(ious)
# instead of using default 0 as threshold, use validation data to find the best threshold.
threshold_best_index = np.argmax(ious) 
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
plt.legend()
for i in range(1,40):
    plot_mosaic(i,1,0)
t1 = time.time()
pred_dict = {idx: rle_encode(np.round(downsample(preds_test[i]) > threshold_best)) for i, idx in enumerate(tqdm_notebook(test_df.index.values))}
t2 = time.time()

print(f"Usedtime = {t2-t1} s")
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv(submission_file)
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")

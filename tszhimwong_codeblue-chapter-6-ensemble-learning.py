# import system modules
import os
import sys
import datetime
import random

# import external helpful libraries
import tensorflow as tf
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa

# import keras
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras.layers import Input, UpSampling2D, concatenate  
from keras.optimizers import Nadam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard

# possible libraries for metrics
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

#K-Fold Cross Validation
import sklearn
from sklearn.model_selection import train_test_split

# Set the random seed to ensure reproducibility
np.random.seed(1234)
tf.random.set_seed(1234)
# Loading our training and validating datasets 
dataset_path = "../input/gametei2020/dataset"

#Form a full dataset 
def combine_dataset(dataset_path, split1, split2):
    split1_path = os.path.join(dataset_path, split1)
    split2_path = os.path.join(dataset_path, split2)
    data_out = []
    
    # iterate each class
    classes = ["NORMAL", "PNEUMONIA"]
    # notice that class_idx = 0 for NORMAL, 1 for PNEUMONIA
    for class_idx, _class in enumerate(classes):
        class_path1 = os.path.join(split1_path, _class) # path to each class dir
        class_path2 = os.path.join(split2_path, _class)
        # iterate through all files in dir
        for filename in os.listdir(class_path1):
            # ensure files are images, if so append to output
            if filename.endswith(".jpeg"):
                img_path = os.path.join(class_path1, filename)
                data_out.append((img_path, class_idx))
        for filename in os.listdir(class_path2):
            # ensure files are images, if so append to output
            if filename.endswith(".jpeg"):
                img_path = os.path.join(class_path2, filename)
                data_out.append((img_path, class_idx))
                
    return data_out
dataset_seq = combine_dataset(dataset_path,split1 = "train",split2 = "val")
dataset_pneumonia_cases = sum([class_idx for (img_path, class_idx) in dataset_seq])
dataset_normal_cases = len(dataset_seq) - dataset_pneumonia_cases
print("Combined - Total: %d, Normal: %d, Pneumonia: %d" % (len(dataset_seq), dataset_normal_cases, dataset_pneumonia_cases))
# Loading the Data Generator
class xray_data_generator(keras.utils.Sequence):
    """
    Data generator derived from Keras' Sequence to be used with fit_generator.
    """
    def __init__(self, seq, dims=(331,331), batch_size=32, shuffle=True):
        # Save params into self
        self.dims = dims
        self.batch_size = batch_size
        self.seq = seq
        self.shuffle = shuffle
        
        # create data augmentor
        self.aug = iaa.SomeOf((0,3),[
                #iaa.Fliplr(), # horizontal flips
                iaa.Affine(scale={"x": (0.75, 1.25), "y": (0.75, 1.25)}),
                iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),    
                iaa.Affine(rotate=(-10, 10)), # rotate images
                iaa.Multiply((0.8, 1.2)),  #random brightness
                iaa.Affine(shear=(-10, 10)),
                #iaa.GammaContrast((0.8, 1.2)),
                iaa.GaussianBlur(sigma=(0.0, 1.0))
                                    ],
                random_order=True
                             )

        # shuffle the dataset
        if self.shuffle:
          random.shuffle(self.seq)    

    def get_data(self, index):
        '''
        Given an index, retrieve the image and apply processing,
        including resizing and converting color encoding. This is
        where data augmentation can be added if desired.
        '''
        img_path, class_idx = self.seq[index]
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.dims)

        # if grayscale, convert to RGB
        if img.shape[-1] == 1:
            img = np.stack((img,img,img), axis=-1)

        # by default, cv2 reads images in using BGR format
        # we want to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize values to [0, 1]
        img = img.astype(np.float32)/255.

        # augment image
        img = self.aug.augment_image(img)

        # Load the labels
        label = keras.utils.to_categorical(class_idx, num_classes=2)
        
        return img, label
      
    def get_classes(self):
        class_idxs = [class_idx for _, class_idx in self.seq]
        return np.array(class_idxs)

    def __len__(self):
        '''
        Returns the number of batches per epoch.
        Used by Keras' fit_generator to determine number of training steps.
        '''
        return int(np.floor(len(self.seq) / self.batch_size))

    def __getitem__(self, index):
        '''
        Actual retrieval of batch data during training.
        Data is retrieved by calling self.get_data on an index
        which is then batched, and returned
        '''
        # create empty batches
        batch_img = np.empty((self.batch_size,) + self.dims + (3,))
        batch_label = np.empty((self.batch_size,) + (2,))

        # load the images and labels into the batch
        # using the get_data method defined above
        for i in range(self.batch_size):
            img, label = self.get_data(index*self.batch_size+i)    
            batch_img[i] = img
            batch_label[i] = label

        return batch_img, batch_label

    def on_epoch_end(self):
        '''
        Shuffles the data sequence after each epoch
        '''
        if self.shuffle:
          random.shuffle(self.seq)
# Loading the customized functions
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
#Ensembling Xception Model
filename = '../input/pneumoniamodels/best_Xception_final.h5'
# load model from file
xception = keras.models.load_model(filename, custom_objects={'focal_crossentropy': focal_loss})
print('>loaded %s' % filename)

#Ensembling InceptionResNetV2 Model
filename = '../input/pneumoniamodels/best_InceptionResNetV2_final.h5'
# load model from file
irn = keras.models.load_model(filename, custom_objects={'focal_crossentropy': focal_loss})
print('>loaded %s' % filename)

#Ensembling InceptionV3 Model
filename = '../input/pneumoniamodels/best_InceptionV3_final.h5'
# load model from file
inception = keras.models.load_model(filename, custom_objects={'focal_crossentropy': focal_loss})
print('>loaded %s' % filename)

#Ensembling DenseNet201 Model
filename = '../input/pneumoniamodels/best_DenseNet201_final.h5'
# load model from file
dense = keras.models.load_model(filename, custom_objects={'focal_crossentropy': focal_loss})
print('>loaded %s' % filename)

# Create a validation generator that does not shuffle
# This will allow our predicted value to match our true values in sequence
noshuf_dataset_gen = xray_data_generator(dataset_seq, batch_size=2, shuffle=False)
trues = noshuf_dataset_gen.get_classes() # True Values

xception_raw = np.array([])
irn_raw = np.array([])
inception_raw = np.array([])
dense_raw = np.array([])
xception_raw = xception.predict_generator(noshuf_dataset_gen) # Predicted values
print("Predicted: Xception")
irn_raw = irn.predict_generator(noshuf_dataset_gen) # Predicted values
print("Predicted: InceptionResNetV2")      
inception_raw = inception.predict_generator(noshuf_dataset_gen) # Predicted values
print("Predicted: InceptionV3")
dense_raw = dense.predict_generator(noshuf_dataset_gen) # Predicted values
print("Predicted: DenseNet201")

# Grid search for Best voting weight
best_auc = 0 # Initiate best AUC
for a in range (1,10):
    for b in range (1,10):
        for c in range (1,10):
            for d in range (1,10):
                #Scales the predictions to find out the best AUC 
                xception_preds = xception_raw 
                xception_preds = np.argmax(xception_preds, axis = 1)
                irn_preds = irn_raw 
                irn_preds= np.argmax(irn_preds, axis = 1)
                inception_preds = inception_raw 
                inception_preds = np.argmax(inception_preds, axis = 1)
                dense_preds = dense_raw 
                dense_preds = np.argmax(dense_preds, axis = 1)
                x = np.zeros(len(dataset_seq))
                y = np.zeros(len(dataset_seq))
                out = np.zeros(len(dataset_seq))
                for i in range(len(dataset_seq)):
                    if xception_preds[i] == 0:
                        x[i] += a 
                    if xception_preds[i] == 1:
                        y[i] += a 
                    if irn_preds[i] == 0:
                        x[i] += b 
                    if irn_preds[i] == 1:
                        y[i] += b 
                    if irn_preds[i] == 0:
                        x[i] += c                 
                    if inception_preds[i] == 1:
                        y[i] += c 
                    if dense_preds[i] == 0:
                        x[i] += d 
                    if dense_preds[i] == 1:
                        y[i] += d 
                    if x[i] > y[i]:
                        out[i] = 0
                    if x[i] <= y[i]:
                        out[i] = 1

                auc = roc_auc_score(trues,out)
                
                if auc > best_auc:
                    best_a = a
                    best_b = b
                    best_c = c
                    best_d = d
                    best_auc = auc

print("Best Voting Weight: Xception %d,InceptionResNet %d,Inception %d, DenseNet %d; Best AUC: %.10f" % (best_a,best_b,best_c,best_d,best_auc))
# Output the result from voting ensemble 
dataset_path = "../input/gametei2020/dataset"

# Define batch size
batch_size = 32    
# Get model input shape by accessing the layers of the model
# e.g. (224,224)
model_input_shape =  (331,331,3)
# Obtain dimensions (width, height)
# e.g. (224,224)
dims = (331,331)
# Get path to the test split in the dataset
test_folder_path = os.path.join(dataset_path, "test")

# Evaluate the model in batches
outs = np.array([])   # compile all the outputs
# iterate through all files in test dir
# sorting ensures that we have a standardised order of predictions
# i.e. always give predictions in order: (0000.jpeg, 0001.jpeg,...)
files = sorted(os.listdir(test_folder_path)) 
# get only images
image_filter_lambda = lambda x: x.endswith(".jpeg")   
imgs = list(filter(image_filter_lambda, files))

print("Making predictions...")
num_imgs = len(imgs)
# Get number of batches to compute
num_batches = int(num_imgs / batch_size) + 1
# tqdm gives us a progress bar
for batch_idx in tqdm(range(num_batches), file=sys.stdout):
    # compute the start and end index for each batch
    start = batch_idx * batch_size
    if batch_idx == num_batches - 1 and (batch_idx + 1) * batch_size > num_imgs:
        end = num_imgs  # clip our end index to the number of images
    else:
        end = (batch_idx + 1) * batch_size

    # create empty batch
    batch_img = np.empty((end - start,) + model_input_shape)
    # load images into batch
    for i in range(end - start):

        img_path = os.path.join(test_folder_path, imgs[start+i])
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, dims)

        # if grayscale, convert to RGB
        if img.shape[-1] == 1:
            img = np.stack((img,img,img), axis=-1)

        # by default, cv2 reads images in using BGR format
        # we want to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize values to [0, 1]
        img = img.astype(np.float32)/255.  
        batch_img[i] = img

    
    # Make prediction for batch
    xception_batch = xception.predict(x=batch_img) # Predicted values
    irn_batch = irn.predict(x=batch_img) # Predicted values  
    inception_batch = inception.predict(x=batch_img) # Predicted values
    dense_batch = dense.predict(x=batch_img) # Predicted values
    
    xception_preds = xception_batch 
    xception_preds = np.argmax(xception_preds, axis = 1)
    irn_preds = irn_batch
    irn_preds= np.argmax(irn_preds, axis = 1)
    inception_preds = inception_batch 
    inception_preds = np.argmax(inception_preds, axis = 1)
    dense_preds = dense_batch 
    dense_preds = np.argmax(dense_preds, axis = 1)
    
    x = np.zeros(end-start)
    y = np.zeros(end-start)
    out = np.zeros(end-start)
    for i in range(end-start):
        if xception_preds[i] == 0:
            x[i] += best_a 
        if xception_preds[i] == 1:
            y[i] += best_a 
        if irn_preds[i] == 0:
            x[i] += best_b 
        if irn_preds[i] == 1:
            y[i] += best_b 
        if irn_preds[i] == 0:
            x[i] += best_c                 
        if inception_preds[i] == 1:
            y[i] += best_c 
        if dense_preds[i] == 0:
            x[i] += best_d 
        if dense_preds[i] == 1:
            y[i] += best_d 
        if x[i] > y[i]:
            out[i] = 0
        if x[i] <= y[i]:
            out[i] = 1
    
    # Scaling of the predictions
    # argmax gives us the class predicted from probabilities
    # e.g. probabilities (0.3, 0.7) -> class = 1 (PNEUMONIA) predicted        
    # out = np.argmax(out, axis=-1)
    outs = np.append(outs, out)
    print("Loaded batch: %d"% batch_idx)
    
# convert into integers (0.0 -> 0, 1.0 -> 1)
outs = outs.astype(int)

# output predictions as .csv file
output_df = pd.DataFrame(outs)
output_df.columns = ['Prediction']
output_df['Id'] = output_df.index
output_df = output_df[['Id', 'Prediction']]
print(output_df)
output_df.to_csv("voting.csv", index=False)
print("Output csv saved")
#Sum Ensemble
n_folds = 4

preds0 = xception_raw[:,0] #Value for normal 
preds0 = np.vstack((preds0,irn_raw[:,0])) #Value for normal 
preds0 = np.vstack((preds0,inception_raw[:,0] )) #Value for normal 
preds0 = np.vstack((preds0,dense_raw[:,0])) #Value for normal 

preds1 = xception_raw[:,1] #Value for Pneumonia 
preds1 = np.vstack((preds1,irn_raw[:,1])) #Value for Pneumonia 
preds1 = np.vstack((preds1,inception_raw[:,1])) #Value for Pneumonia 
preds1 = np.vstack((preds1,dense_raw[:,1])) #Value for Pneumonia  

out = np.zeros(len(dataset_seq))
sum_preds0 = np.zeros(len(dataset_seq))
sum_preds1 = np.zeros(len(dataset_seq))
for i in range(len(dataset_seq)):
    sum_preds0[i] = preds0[0][i]+preds0[1][i]+preds0[2][i]+preds0[3][i]
    sum_preds1[i] = preds1[0][i]+preds1[1][i]+preds1[2][i]+preds1[3][i]
    if sum_preds0[i] < sum_preds1[i]:
        out[i] = 1
    else:
        out[i] = 0 

# Compute metrics
acc = accuracy_score(trues, out)
prec = precision_score(trues, out)
rec = recall_score(trues, out)
f1 = f1_score(trues, out)
auc = roc_auc_score(trues, out)

# Print metrics summary
print("Evaluation of Sum Ensemble Model")
print("Accuracy: %.3f" % acc)
print("Precision: %.3f" % prec)
print("Recall: %.3f" %  rec)
print("F1: %.3f" % f1)
print("AUC: %.3f" % auc)


from sklearn.metrics import confusion_matrix
results = confusion_matrix(trues, out)
print(results)

# Preprocess the feature dataframe and the target

features = []
targets = []
features = xception_raw[:,0]
features = np.vstack((features,xception_raw[:,1]))
features = np.vstack((features,irn_raw[:,0]))
features = np.vstack((features,irn_raw[:,1]))
features = np.vstack((features,inception_raw[:,0]))
features = np.vstack((features,inception_raw[:,1]))
features = np.vstack((features,inception_raw[:,0]))
features = np.vstack((features,inception_raw[:,1]))
features = features.transpose()

targets = trues
#target = target.reshape(-1,1)
print(features.shape)
print(targets.shape)
# We sample 20% of the data as validation set
train_idx = random.sample(range(len(dataset_seq)),int(len(dataset_seq)*0.8))
train_idx = np.array(train_idx)
val_idx = np.delete(range(len(dataset_seq)),train_idx)

train_features = features[train_idx]
train_targets = targets[train_idx]
val_features = features[val_idx]
val_targets = targets[val_idx]
print(train_features.shape)
print(train_targets.shape)
print(val_features.shape)
print(val_targets.shape)
#Construct the 1-layer model
from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.models import Model
from keras import backend as K
ensemble = models.Sequential()
ensemble.add(keras.Input(shape=features.shape[-1]))
#ensemble.add(layers.Dense(8, activation="relu"))
#ensemble.add(layers.Dense(8, activation="relu"))
#ensemble.add(layers.Dense(8, activation = "relu"))
ensemble.add(layers.Dense(1, activation="sigmoid"))
ensemble.summary()
ensemble.compile(optimizer = keras.optimizers.Adam(lr=0.01),
              loss = 'binary_crossentropy',
              metrics='AUC')
checkpt = ModelCheckpoint(monitor='val_loss',
                        filepath = 'best_Ensemble_final.h5',
                        mode = 'min',
                        save_best_only=True)
weight_for_0 = dataset_pneumonia_cases / len(dataset_seq)
weight_for_1 = dataset_normal_cases / len(dataset_seq)
class_weight = {0: weight_for_0, 1: weight_for_1}
history = ensemble.fit(train_features, train_targets,
          batch_size=32,
          epochs=500,
          verbose=2,
          callbacks=checkpt,
          validation_data=(val_features, val_targets),
          class_weight=class_weight
)
# Evaluate the ensemble neural network model
ensemble = keras.models.load_model("best_Ensemble_final.h5")
out = ensemble.predict(features)
out = (out > 0.5).astype(int)


# Compute metrics
acc = accuracy_score(trues, out)
prec = precision_score(trues, out)
rec = recall_score(trues, out)
f1 = f1_score(trues, out)
auc = roc_auc_score(trues, out)

# Print metrics summary
print("Evaluation of model combined model")
print("Accuracy: %.3f" % acc)
print("Precision: %.3f" % prec)
print("Recall: %.3f" %  rec)
print("F1: %.3f" % f1)
print("AUC: %.3f" % auc)


from sklearn.metrics import confusion_matrix
results = confusion_matrix(trues, out)
print(results)

#Neural Network Ensemble 
dataset_path = "../input/gametei2020/dataset"

# Define batch size
batch_size = 32    
# Get model input shape by accessing the layers of the model
# e.g. (224,224)
model_input_shape =  (331,331,3)
# Obtain dimensions (width, height)
# e.g. (224,224)
dims = (331,331)
# Get path to the test split in the dataset
test_folder_path = os.path.join(dataset_path, "test")

# Evaluate the model in batches
outs = np.array([])   # compile all the outputs
# iterate through all files in test dir
# sorting ensures that we have a standardised order of predictions
# i.e. always give predictions in order: (0000.jpeg, 0001.jpeg,...)
files = sorted(os.listdir(test_folder_path)) 
# get only images
image_filter_lambda = lambda x: x.endswith(".jpeg")   
imgs = list(filter(image_filter_lambda, files))

print("Making predictions...")
num_imgs = len(imgs)
# Get number of batches to compute
num_batches = int(num_imgs / batch_size) + 1
# tqdm gives us a progress bar
for batch_idx in tqdm(range(num_batches), file=sys.stdout):
    # compute the start and end index for each batch
    start = batch_idx * batch_size
    if batch_idx == num_batches - 1 and (batch_idx + 1) * batch_size > num_imgs:
        end = num_imgs  # clip our end index to the number of images
    else:
        end = (batch_idx + 1) * batch_size

    # create empty batch
    batch_img = np.empty((end - start,) + model_input_shape)
    # load images into batch
    for i in range(end - start):

        img_path = os.path.join(test_folder_path, imgs[start+i])
        # Load the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, dims)

        # if grayscale, convert to RGB
        if img.shape[-1] == 1:
            img = np.stack((img,img,img), axis=-1)

        # by default, cv2 reads images in using BGR format
        # we want to convert it to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize values to [0, 1]
        img = img.astype(np.float32)/255.  
        batch_img[i] = img

    
    xception_batch = xception.predict(x=batch_img) # Predicted values
    irn_batch = irn.predict(x=batch_img) # Predicted values  
    inception_batch = inception.predict(x=batch_img) # Predicted values
    dense_batch = dense.predict(x=batch_img) # Predicted values
    

    features = []
    features = xception_batch[:,0]
    features = np.vstack((features,xception_batch[:,1]))
    features = np.vstack((features,irn_batch[:,0]))
    features = np.vstack((features,irn_batch[:,1]))
    features = np.vstack((features,inception_batch[:,0]))
    features = np.vstack((features,inception_batch[:,1]))
    features = np.vstack((features,dense_batch[:,0]))
    features = np.vstack((features,dense_batch[:,1]))
    features = features.transpose()
    
    ensemble = keras.models.load_model("best_Ensemble_final.h5")
    out = ensemble.predict(features)
    out = (out > 0.5).astype(int)
    
    # Scaling of the predictions
    # argmax gives us the class predicted from probabilities
    # e.g. probabilities (0.3, 0.7) -> class = 1 (PNEUMONIA) predicted        
    # out = np.argmax(out, axis=-1)
    outs = np.append(outs, out)
    print("Loaded batch: %d"% batch_idx)
    
# convert into integers (0.0 -> 0, 1.0 -> 1)
outs = outs.astype(int)

# output predictions as .csv file
output_df = pd.DataFrame(outs)
output_df.columns = ['Prediction']
output_df['Id'] = output_df.index
output_df = output_df[['Id', 'Prediction']]
print(output_df)
output_df.to_csv("neuralens.csv", index=False)
print("Output csv saved")
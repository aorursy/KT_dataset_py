# !conda install -c conda-forge gdcm -y
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import random
import pathlib
import pickle
import time
import copy
import pydicom
import matplotlib.pyplot as plt
import h5py
import scipy

# import tensorflow.keras.backend
# from tensorflow.keras import layers as L
from skimage.transform import resize
from scipy.ndimage import zoom
import scipy.ndimage as ndimage
from skimage import measure, morphology, segmentation

from math import ceil
# raw_train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')
train = pd.read_csv('/kaggle/input/data-preparation-for-osic/train.csv')

# size_image = pd.read_csv('/kaggle/input/prep-data/size_image.csv')
# list_files = pd.read_csv('/kaggle/input/prep-data/list_files.csv', converters={"files": lambda x: x.strip("[]").replace("'","").split(", ")})
#Constant
TRAIN_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/train'
TEST_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

PINBALL_QUANTILE = np.array([0.2, 0.50, 0.8])
LAMBDA_LOSS = 0.75
ID='Patient_Week'

DESIRED_SIZE = (50,512,512)
BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
MASK_ITERATION = 4

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556

PROB_DROPOUT = 0.2
# start_time = time.perf_counter()
# for i in train.loc['Patient'].unique():
#     file = h5py.File('/kaggle/input/creating-dataset-2/'+ i +'.h5', 'r')
#     np.array(file.get('data'))[0,0,:,:,0]
#     tf.print("End epoch:", time.perf_counter() - start_time)
#     file.close

test['Min_week'] = test.groupby('Patient')['Weeks'].transform('min')
base = test.loc[test.Weeks == test.Min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','Base_FVC']
test = test.merge(base, on='Patient', how='left')
test['Base_week'] = test['Weeks'] - test['Min_week']
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder

#Override OneHotEncoder to have the column names created automatically (Ex-smoker, Never Smoked...) 
class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index='', name='', **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append('{}_{}'.format(name, categories[j]))
        return new_columns

from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

def standardisation(x, u, s):
    return (x-u)/s

def normalization(x, ma, mi):
    return (x-mi)/(ma-mi)

class data_preparation():
    def __init__(self, bool_normalization=True,bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization
        
        
    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)
        
        #For the test set/Already fitted
        try:
            data['Sex'] = self.enc_sex.transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            #Standardisation
            if self.standardisation:
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)
            
            #Normalization
            if self.normalization:
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

        #For the train set/Not yet fitted    
        except NotFittedError:
            data['Sex'] = self.enc_sex.fit_transform(data['Sex'].values)
            data['SmokingStatus'] = self.enc_smok.fit_transform(data['SmokingStatus'].values)
            data = pd.concat([data.drop(columns=['SmokingStatus']), self.onehotenc_smok.fit_transform(data['SmokingStatus'].values.reshape(-1,1), categories=self.enc_smok.classes_, name='', index=data.index).astype(int)], axis=1)
            
            #Standardisation
            if self.standardisation:
                self.base_week_mean = data['Base_week'].mean()
                self.base_week_std = data['Base_week'].std()
                data['Base_week'] = standardisation(data['Base_week'],self.base_week_mean,self.base_week_std)

                self.base_fvc_mean = data['Base_FVC'].mean()
                self.base_fvc_std = data['Base_FVC'].std()
                data['Base_FVC'] = standardisation(data['Base_FVC'],self.base_fvc_mean,self.base_fvc_std)

                self.base_percent_mean = data['Base_percent'].mean()
                self.base_percent_std = data['Base_percent'].std()
                data['Base_percent'] = standardisation(data['Base_percent'],self.base_percent_mean,self.base_percent_std)

                self.age_mean = data['Age'].mean()
                self.age_std = data['Age'].std()
                data['Age'] = standardisation(data['Age'],self.age_mean,self.age_std)

                self.weeks_mean = data['Weeks'].mean()
                self.weeks_std = data['Weeks'].std()
                data['Weeks'] = standardisation(data['Weeks'],self.weeks_mean,self.weeks_std)

                
            #Normalization
            if self.normalization:
                self.base_week_min = data['Base_week'].min()
                self.base_week_max = data['Base_week'].max()
                data['Base_week'] = normalization(data['Base_week'],self.base_week_max,self.base_week_min)

                self.base_fvc_min = data['Base_FVC'].min()
                self.base_fvc_max = data['Base_FVC'].max()
                data['Base_FVC'] = normalization(data['Base_FVC'],self.base_fvc_max,self.base_fvc_min)

                self.base_percent_min = data['Percent'].min()
                self.base_percent_max = data['Percent'].max()
                data['Percent'] = normalization(data['Percent'],self.base_percent_max,self.base_percent_min)

                self.age_min = data['Age'].min()
                self.age_max = data['Age'].max()
                data['Age'] = normalization(data['Age'],self.age_max,self.age_min)

                self.weeks_min = data['Weeks'].min()
                self.weeks_max = data['Weeks'].max()
                data['Weeks'] = normalization(data['Weeks'],self.weeks_max,self.weeks_min)
                
                self.base_week_min = data['Min_week'].min()
                self.base_week_max = data['Min_week'].max()
                data['Min_week'] = normalization(data['Min_week'],self.base_week_max,self.base_week_min)

            
        return data
pickefile = open('/kaggle/input/data-preparation-for-osic/data_prep', 'rb')
data_prep = pickle.load(pickefile)
pickefile.close()
test = data_prep(test)
class DataGenerator(tf.keras.utils.Sequence):
    
    def on_epoch_end(self):#Indices=np.arange(size of dataset)
        self.indices = np.arange(len(self.patients))
    
    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))
    
    def __init__(self, train, patients, batch_size=1, desired_size=(10,512,512), img_path=TRAIN_PATH, *args, **kwargs):
        self.train = train
        self.patients = patients
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        patients = [self.patients[k] for k in indices]
#         imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
        if 0<=index<=47:
            notebook = '1'
        elif 48<=index<=95:
            notebook = '2'
        elif 96<=index<=142:
            notebook = '3'
        elif 143<=index<=177:
            notebook = '4'
        file = h5py.File('/kaggle/input/creating-dataset-'+ notebook +'/'+ patients[0] +'.h5', 'r')
        imgs = np.array(file.get('data'))
        file.close()
        
        
#         Data Augmentation
#         Rotation
        rand = np.random.randint(1,20)
        if rand == 1:
            print('Rotate1')
            imgs = scipy.ndimage.rotate(imgs, np.random.randint(10,180), axes=(2,3), reshape=False)
        elif rand == 2:
            print('Rotate2')
            imgs = scipy.ndimage.rotate(imgs, np.random.randint(-180,-10), axes=(2,3), reshape=False)
        
        #Shift
        if np.random.randint(1,10) == 1:
            print('Shift')
            imgs = scipy.ndimage.shift(imgs, [0, np.random.randint(-8,8), np.random.randint(-64,64), np.random.randint(-64,64), 0], order=0)
        
        #Flip
        if  np.random.randint(1,20) == 1:
            print('Flip')
            imgs = np.flip(imgs, axis = np.unique(np.random.randint(2,4, size=np.random.randint(1,3))))
        
#         if  np.random.randint(1,3) == 1:  
#             imgs = zoom(imgs, [1,np.random.random()+1,np.random.random()+1,np.random.random()+1,1], mode='nearest')
#         #Contrast
#         factor=np.random.randint(-3, 3)/1000
#         if factor != 0:
#             imgs = 128/255 + factor * imgs - factor * 128/255
        
        #Zoom
#         if  np.random.randint(1,20) == 1:
#             print('Zoom1')
#             factor = min(round(np.random.random()+1, 2), 1.5)
#             shape = imgs.shape[3]
#             if factor>1:
#                 imgs = zoom(imgs, [1, 1, 1, factor,1], mode='nearest', order=1)
#                 imgs = imgs[:,:,:,round((imgs.shape[3]-shape-1)/2):-round((imgs.shape[3]-shape-1)/2),:][:,:,:,:DESIRED_SIZE[2],:]
                
#         if  np.random.randint(1,20) == 1:
#             print('Zoom2')
#             factor = min(round(np.random.random()+1, 2), 1.5)
#             shape = imgs.shape[2]
#             if factor>1:
#                 imgs = zoom(imgs, [1, 1, factor, 1,1], mode='nearest', order=1)
#                 imgs = imgs[:,:,round((imgs.shape[2]-shape-1)/2):-round((imgs.shape[2]-shape-1)/2),:,:][:,:,:DESIRED_SIZE[1],:,:]
                
#         if  np.random.randint(1,20) == 1:
#             print('Zoom3')
#             factor = min(round(np.random.random()+1, 2), 1.5)
#             shape = imgs.shape[1]
#             if factor>1:
#                 imgs = zoom(imgs, [1, factor, 1, 1,1], mode='nearest', order=1)
#                 imgs = imgs[:,round((imgs.shape[1]-shape-1)/2):-round((imgs.shape[1]-shape-1)/2),:,:,:][:,:DESIRED_SIZE[0],:,:,:]
                
        return [self.train[self.train['Patient'].isin(patients)].reset_index(drop=True), imgs], np.asarray(self.train[self.train['Patient'].isin(patients)]['FVC'])   
train_generator = DataGenerator(train, train.Patient.unique(), batch_size=BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TRAIN_PATH)
test_generator = DataGenerator(test, test.Patient.unique(), batch_size=TEST_BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TEST_PATH)
# for [X1, X2], Y in train_generator:
#     if X2.shape != (1,50,512,512,1):
#         print(X2.shape)
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
#=============================#
def score(y_true, y_pred):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return tf.keras.backend.mean(metric)
#============================#
def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = PINBALL_QUANTILE
    q = tf.constant(PINBALL_QUANTILE, dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return tf.keras.backend.mean(v)
#=============================#
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss
# MLP = tf.keras.models.load_model('/kaggle/input/simple-mlp-training/model', compile=False, custom_objects={'score':score})
image_input = tf.keras.Input(shape=(DESIRED_SIZE[0],DESIRED_SIZE[1],DESIRED_SIZE[2],1) ,name="img_input")
x = tf.keras.layers.Conv3D(filters=8, kernel_size=(1,3,3), padding='valid', activation='relu')(image_input)
x = tf.keras.layers.Conv3D(filters=8, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2))(x)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=32, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(1,2,2),strides=(1,2,2))(x)
x = tf.keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=64, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(x)
x = tf.keras.layers.Conv3D(filters=256, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=256, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(x)
x = tf.keras.layers.Conv3D(filters=512, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=512, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.Conv3D(filters=512, kernel_size=(1,3,3), padding='valid', activation='relu')(x)
x = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),strides=(2,2,2))(x)
x = tf.keras.layers.GlobalMaxPooling3D()(x)
cnn_output= tf.keras.layers.Dense(100, activation='relu')(x)


numpy_input = tf.keras.Input(shape=(10,), name="numpy_input")
# x = MLP.get_layer('dense_to_freeze1')(numpy_input)
# x = MLP.get_layer('dense_to_freeze2')(x)
x = tf.keras.layers.Dense(500, activation='relu', name='dense_to_freeze1')(numpy_input)
x = tf.keras.layers.Dense(100, activation='relu', name='dense_to_freeze2')(x)
x = tf.keras.layers.GaussianNoise(0.2, name='GaussianNoise')(x)
x = tf.keras.layers.concatenate([cnn_output, x], axis=1, name='concatenate')
x = tf.keras.layers.Dropout(PROB_DROPOUT, name='Dropout')(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(20, activation='relu')(x)
output= tf.keras.layers.Dense(3)(x)
CNN = tf.keras.Model(inputs=image_input, outputs=cnn_output)
model = tf.keras.Model(inputs=[numpy_input, CNN.input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999), loss=mloss(LAMBDA_LOSS))
    
# model.get_layer('dense_to_freeze1').trainable=False
# model.get_layer('dense_to_freeze2').trainable=False
tf.keras.utils.plot_model(model, show_shapes=True)
def eval_score(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    #sigma_clip = sigma + C1
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return -tf.keras.backend.mean(metric)
model = tf.keras.models.load_model('/kaggle/input/3d-cnn-mlp/model_4', custom_objects={'loss':mloss(LAMBDA_LOSS), 'score':score})
CNN = tf.keras.models.load_model('/kaggle/input/3d-cnn-mlp/CNN_4', custom_objects={'loss':mloss(LAMBDA_LOSS), 'score':score})
# pre_trained_model = tf.keras.models.load_model('/kaggle/input/3d-cnn-mlp/model_9', custom_objects={'loss':mloss(LAMBDA_LOSS), 'score':score})
# pre_trained_CNN = tf.keras.models.load_model('/kaggle/input/3d-cnn-mlp/CNN_9', custom_objects={'loss':mloss(LAMBDA_LOSS), 'score':score})
# CNN.set_weights(pre_trained_CNN.get_weights())
# model.set_weights(pre_trained_model.get_weights())
# for i in range(len(CNN.layers)):
#     CNN.layers[i].set_weights(pre_trained_CNN.layers[i].get_weights())
#     CNN.layers[i].trainable = False
def train_step(X1, X2, Y, j, list_loss):
    with tf.GradientTape() as tape:
        X2 = tf.convert_to_tensor(X2)
        tape.watch(X2)
        out_imgs = CNN(X2)
        X_imgs = tf.concat([tf.repeat(tf.reshape(out_imgs[i], (1,-1)), X1['Patient'].value_counts()[i], axis=0) for i in range(len(X1['Patient'].value_counts()))], axis=0)
        X1 = tf.convert_to_tensor(np.asarray(X1[SELECTED_COLUMNS]))
        tape.watch(X1)
        X_tabular = model.get_layer('GaussianNoise')(model.get_layer('dense_to_freeze2')(model.get_layer('dense_to_freeze1')(X1)), training=True)
        inp_mlp = model.get_layer('Dropout')(model.get_layer('concatenate')([X_imgs, X_tabular]), training=True)
        inp_mlp = model.layers[-1](model.layers[-2](model.layers[-3](inp_mlp)))
        loss = model.compiled_loss(tf.constant(Y), inp_mlp, regularization_losses=model.losses)
    print('loss calculated')
    grads = tape.gradient(loss, model.trainable_weights)
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

#     if loss > 30:
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#   if loss > 40:
#       model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#       print('Optimized 2 times')
#   if loss > 50:
#       model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#       print('Optimized 3 times')
#     if loss > 60:
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         print('Optimized 5 times')
    list_loss.append(loss)
            
    print("Training loss (for one batch) at step %d: %.4f" % (j, float(loss)))
    return list_loss
    

def eval_step():
    y_eval_pred = np.array([[0,0,0]])
    y_eval_true = np.array([0])
    for [X1_eval, X2_eval], Y_eval in test_generator:
        X2_eval = tf.convert_to_tensor(X2_eval)
        out_imgs = CNN(X2_eval)
        X_imgs = tf.concat([tf.repeat(tf.reshape(out_imgs[i], (1,-1)), X1_eval['Patient'].value_counts()[i], axis=0) for i in range(len(X1_eval['Patient'].value_counts()))], axis=0)
        X1_eval = tf.convert_to_tensor(np.asarray(X1_eval[SELECTED_COLUMNS]))
        X_tabular = model.get_layer('dense_to_freeze2')(model.get_layer('dense_to_freeze1')(X1_eval))
        inp_mlp = model.get_layer('concatenate')([X_imgs, X_tabular])
        inp_mlp = model.layers[-1](model.layers[-2](model.layers[-3](inp_mlp)))
        y_eval_pred = np.append(y_eval_pred, inp_mlp.numpy(), axis=0)
        y_eval_true = np.append(y_eval_true, Y_eval)
    y_eval_pred = y_eval_pred[1:]
    y_eval_true = y_eval_true[1:]
    sc = eval_score(y_eval_true, y_eval_pred)
    tf.print('Eval Score:%f ' % sc)
SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']
epochs = 5
loss = []
for epoch in range(epochs):
    list_loss = []
    start_time = time.perf_counter()
    print("\nStart of epoch %d" % (epoch,))
    j=0
    for [X1, X2], Y in train_generator:
        list_loss = train_step(X1, X2, Y, j, list_loss)
        j+=1
    loss.append(np.mean(list_loss))
    print(np.mean(list_loss))
    if epoch%1==0:
        CNN.save('CNN_' + str(epoch))
        model.save('model_' + str(epoch))
    tf.print("End epoch:", time.perf_counter() - start_time)
#     eval_step()
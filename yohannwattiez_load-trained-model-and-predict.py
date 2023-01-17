import tensorflow as tf

import pandas as pd

import numpy as np

import os

import random

import pathlib

import pickle

import time



import pydicom

import matplotlib.pyplot as plt



import tensorflow.keras.backend

from tensorflow import keras as K

from tensorflow.keras import layers as L

from skimage.transform import resize

from scipy.ndimage import zoom

import scipy.ndimage as ndimage

from skimage import measure, morphology, segmentation



from math import ceil
raw_test = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv')

X_prediction = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
#Constant

TEST_PATH = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'



#For data prep

DESIRED_SIZE = (30,256,256)

BATCH_SIZE = 256



MASK_ITERATION = 4



clip_bounds = (-1000, 200)

pre_calculated_mean = 0.02865046213070556
# def create_submission(value):

#     if value==0:

#         sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[0 for i in X_prediction.index], 'Confidence': [10000 for i in X_prediction.index]})

#     elif value==1:

#         sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[100 for i in X_prediction.index], 'Confidence': [5000 for i in X_prediction.index]})

#     elif value==2:

#         sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[500 for i in X_prediction.index], 'Confidence': [1000 for i in X_prediction.index]})

#     elif value==3:

#         sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[1000 for i in X_prediction.index], 'Confidence': [500 for i in X_prediction.index]})

#     elif value==4:

#         sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':[2000 for i in X_prediction.index], 'Confidence': [100 for i in X_prediction.index]})

#     return sub

        
X_prediction['Patient'] = X_prediction['Patient_Week'].str.extract(r'(.*)_.*')

X_prediction['Weeks'] = X_prediction['Patient_Week'].str.extract(r'.*_(.*)').astype(int)

X_prediction = X_prediction[['Patient', 'Weeks', 'Patient_Week']]

rename_cols = {'Weeks_y':'Min_week', 'Weeks_x': 'Weeks', 'FVC':'Base_FVC'}

X_prediction = X_prediction.merge(raw_test, how='left', left_on='Patient', right_on='Patient').rename(columns=rename_cols)[['Patient', 'Min_week', 'Base_FVC', 'Percent', 'Age', 'Sex', 'SmokingStatus', 'Weeks', 'Patient_Week']].reset_index(drop=True)
X_prediction['Base_week'] = X_prediction['Weeks'] - X_prediction['Min_week']
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
X_prediction = data_prep(X_prediction).sort_values('Patient')
class ConvertToHU:

    def __call__(self, imgs, dicom):



#         img_type = dicom.ImageType

#         is_hu = img_type[0] == 'ORIGINAL' and not (img_type[2] == 'LOCALIZER')

        # if not is_hu:

        #     warnings.warn(f'Patient {data.PatientID} CT Scan not cannot be'

        #                   f'converted to Hounsfield Units (HU).')



        intercept = dicom.RescaleIntercept

        slope = dicom.RescaleSlope

        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)

        return imgs

convertohu = ConvertToHU()
class Clip:

    def __init__(self, bounds=(-1000, 500)):

        self.min = min(bounds)

        self.max = max(bounds)



    def __call__(self, image):

        image[image < self.min] = self.min

        image[image > self.max] = self.max

        return image

clip = Clip(clip_bounds)
class MaskWatershed:

    def __init__(self, min_hu, iterations):

        self.min_hu = min_hu

        self.iterations = iterations



    def __call__(self, image, dicom):

        

        # Structuring element used for the filter

        blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

                           [0, 1, 1, 1, 1, 1, 0],

                           [1, 1, 1, 1, 1, 1, 1],

                           [1, 1, 1, 1, 1, 1, 1],

                           [1, 1, 1, 1, 1, 1, 1],

                           [0, 1, 1, 1, 1, 1, 0],

                           [0, 0, 1, 1, 1, 0, 0]]



        blackhat_struct = ndimage.iterate_structure(blackhat_struct, self.iterations)

        stack = []

        for slice_idx in range(image.shape[0]):

            sliced = image[slice_idx]

            stack.append(self.seperate_lungs(sliced, blackhat_struct, self.min_hu,

                                             self.iterations))

        

        

        return np.stack(stack)



    @staticmethod

    def seperate_lungs(image, blackhat_struct, min_hu=min(clip_bounds), iterations=2):

#         h, w = image.shape[0], image.shape[1]



        marker_internal, marker_external, marker_watershed = MaskWatershed.generate_markers(image)

        

        # Sobel-Gradient

        sobel_filtered_dx = ndimage.sobel(image, 1)

        sobel_filtered_dy = ndimage.sobel(image, 0)

        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)

        sobel_gradient *= 255.0 / np.max(sobel_gradient)



        watershed = morphology.watershed(sobel_gradient, marker_watershed)



        outline = ndimage.morphological_gradient(watershed, size=(3,3)).astype(bool)

#         outline = outline.astype(bool)



#         # Structuring element used for the filter

#         blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],

#                            [0, 1, 1, 1, 1, 1, 0],

#                            [1, 1, 1, 1, 1, 1, 1],

#                            [1, 1, 1, 1, 1, 1, 1],

#                            [1, 1, 1, 1, 1, 1, 1],

#                            [0, 1, 1, 1, 1, 1, 0],

#                            [0, 0, 1, 1, 1, 0, 0]]



#         blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)



        # Perform Black Top-hat filter

        outline += ndimage.black_tophat(outline, structure=blackhat_struct)



        lungfilter = np.bitwise_or(marker_internal, outline)

        lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)



        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((image.shape[0], image.shape[1])))



        return segmented  #, lungfilter, outline, watershed, sobel_gradient



    @staticmethod

    def generate_markers(image, threshold=-400):

#         h, w = image.shape[0], image.shape[1]



#         marker_internal = image < threshold

#         marker_internal = segmentation.clear_border(image < threshold)

        marker_internal_labels = measure.label(segmentation.clear_border(image < threshold))

        

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]

        areas.sort()

        

        if len(areas) > 2:

            for region in measure.regionprops(marker_internal_labels):

                if region.area < areas[-2]:

                    for coordinates in region.coords:

                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        

        marker_internal = marker_internal_labels > 0



        # Creation of the External Marker

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)

        external_b = ndimage.binary_dilation(marker_internal, iterations=55)

        marker_external = external_b ^ external_a



        # Creation of the Watershed Marker

        marker_watershed = np.zeros((image.shape[0], image.shape[1]), dtype=np.int)

        marker_watershed += marker_internal * 255

        marker_watershed += marker_external * 128



        return marker_internal, marker_external, marker_watershed

maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=MASK_ITERATION)
class Normalize:

    def __init__(self, bounds=(-1000, 500)):

        self.min = min(bounds)

        self.max = max(bounds)



    def __call__(self, image):

        image = image.astype(np.float)

        image = (image - self.min) / (self.max - self.min)

        return image

     

class ZeroCenter:

    def __init__(self, pre_calculated_mean):

        self.pre_calculated_mean = pre_calculated_mean



    def __call__(self, image):

        return image - self.pre_calculated_mean



normalize = Normalize(bounds=clip_bounds)

zerocenter = ZeroCenter(pre_calculated_mean=pre_calculated_mean)
def sort_function(x): #Get the files in the right order

    return int(x.split('.')[0])



def _read(path, patients=[], desired_size=(60,512,512)):

    X = np.empty(np.concatenate(([len(patients), 1], np.array(desired_size))))

    i=0

    for patient in patients:

        dicom = pydicom.dcmread(path+'/'+patient+'/'+os.listdir(path+'/'+patient)[0])

        list_patient_files = sorted([i for i in os.listdir(path+'/'+patient)], key=sort_function)

#         list_patient_files = list_patient_files[::max(int(len(list_patient_files)/40),1)][:30]

        df = pd.DataFrame(list_patient_files).apply(lambda x:path+'/'+patient+'/'+x)

        df = convertohu(df.iloc[:,0].apply(lambda x: pydicom.dcmread(x).pixel_array), dicom)

        df = zoom(df, np.array(DESIRED_SIZE)/np.array(df.shape), mode='nearest')

#         df = np.array(df.iloc[:,0].apply(lambda x: resize(pydicom.dcmread(x).pixel_array, DESIRED_SIZE[1:3])).to_list())

#         df = zoom(df, np.array(DESIRED_SIZE)/np.array(df.shape), mode='nearest')

        X[i,0,:,:,:] = zerocenter(normalize(maskwatershed(clip(df), dicom)))

        i+=1

            

    #apply on resize on a batch and not on one picture

    return X
# X = _read('/kaggle/input/osic-pulmonary-fibrosis-progression/train', patients=['ID00007637202177411956430'], desired_size=DESIRED_SIZE)
class DataGenerator(K.utils.Sequence):

    

    def on_epoch_end(self):#Indices=np.arange(size of dataset)

        self.indices = np.arange(len(self.list_IDs))

    

    def __len__(self):

        return int(ceil(len(self.indices) / self.batch_size))

    

    def __init__(self, train, list_IDs, batch_size=1, desired_size=(10,512,512), img_path=TEST_PATH, *args, **kwargs):

        self.train = train

        self.list_IDs = list_IDs

        self.batch_size = batch_size

        self.desired_size = desired_size

        self.img_path = img_path

        self.on_epoch_end()

    

    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        

        patients = self.train.loc[list_IDs_temp, 'Patient'].unique()

        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)

        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(imgs, (0, 2, 3, 4, 1))
pred_generator = DataGenerator(X_prediction, X_prediction.index, batch_size=BATCH_SIZE, desired_size=DESIRED_SIZE, img_path=TEST_PATH)
# model = K.models.load_model('/kaggle/input/cnn-for-latent-features/model',custom_objects={'mloss':mloss(LAMBDA_LOSS)})

model = K.models.load_model('/kaggle/input/3d-cnn-mlp/model_4',compile=False)

CNN = K.models.load_model('/kaggle/input/3d-cnn-mlp/CNN_4',compile=False)
# start_time = time.perf_counter()

SELECTED_COLUMNS = ['Weeks', 'Percent', 'Age', 'Sex', 'Min_week', 'Base_FVC','Base_week', '_Currently smokes', '_Ex-smoker', '_Never smoked']

y_prediction = np.array([[0,0,0]])

for X1, X2 in pred_generator:

    out_imgs = CNN(tf.convert_to_tensor(X2))

    X_imgs = tf.concat([tf.repeat(tf.reshape(out_imgs[i], (1,-1)), X1['Patient'].value_counts()[i], axis=0) for i in range(len(X1['Patient'].value_counts()))], axis=0)

    X_tabular = model.get_layer('dense_to_freeze2')(model.get_layer('dense_to_freeze1')(tf.convert_to_tensor(np.asarray(X1[SELECTED_COLUMNS]))))

    inp_mlp = model.get_layer('concatenate')([X_imgs, X_tabular])

    inp_mlp = model.layers[-1](model.layers[-2](model.layers[-3](inp_mlp)))

    y_prediction = np.append(y_prediction, inp_mlp.numpy(), axis=0)

y_prediction = y_prediction[1:]

# tf.print("Execution time:", time.perf_counter() - start_time)
sub = pd.DataFrame(data={'Patient_Week':X_prediction['Patient_Week'], 'FVC':y_prediction[:,1], 'Confidence': (y_prediction[:,2]-y_prediction[:,0])})

sub.to_csv('submission.csv', index=False)
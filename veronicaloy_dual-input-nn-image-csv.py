# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd

import os

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error

import tensorflow as tf
# Regular Imports

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

from scipy.stats import pearsonr



import pydicom # for DICOM images

from skimage.transform import resize

import copy

import re



# Segmentation

from glob import glob

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.transform import resize

from sklearn.cluster import KMeans

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.tools import FigureFactory as FF

from plotly.graph_objs import *

init_notebook_mode(connected=True) 





import warnings

warnings.filterwarnings("ignore")
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

df = pd.read_csv(base_path + 'train.csv')

# df.sample(5,  random_state=1)

df.head()
# Create base director for Train .dcm files

director = "../input/osic-pulmonary-fibrosis-progression/train"



# Create path column with the path to each patient's CT

df["Path"] = director + "/" + df["Patient"]



# Create variable that shows how many CT scans each patient has

df["CT_number"] = 0



for k, path in enumerate(df["Path"]):

    df["CT_number"][k] = len(os.listdir(path))
df = df[df['Patient']!='ID00011637202177653955184']

df = df[df['Patient']!='ID00052637202186188008618']
def get_mid_ct_scan(patient_dir):

    # First Order the files in the dataset

    files = []

    for dcm in list(os.listdir(patient_dir)):

        files.append(dcm) 

    files.sort(key=lambda f: int(re.sub('\D', '', f)))



    # Read the middle image in the Dataset

    mid_ct_scan = len(files)//2

    dcm = files[mid_ct_scan]

    path = patient_dir + "/" + dcm

    datasets = pydicom.dcmread(path)

    img = datasets.pixel_array/2000 #normalize

    img = cv2.resize(img, (224,224))

    #     plt.imshow(img, cmap='plasma')

#     img = img.flatten()



    return img

df['mid_ct_scan'] = df['Path'].apply(lambda x: get_mid_ct_scan(x))
# check img of random patient

img = df['mid_ct_scan'][78]

plt.imshow(img, cmap='plasma')
def get_weeks_passed(df):

    min_week_dict = df.groupby('Patient').min('Weeks')['Weeks'].to_dict()

    df['MinWeek'] =  df['Patient'].map(min_week_dict)

    df['WeeksPassed'] = df['Weeks'] - df['MinWeek']

    return df
def get_baseline_FVC(df):

    _df = (

        df

        .loc[df.Weeks == df.MinWeek][['Patient','FVC']]

        .rename({'FVC': 'FirstFVC'}, axis=1)

        .groupby('Patient')

        .first()

#         .reset_index()

    )

    

    first_FVC_dict = _df.to_dict()['FirstFVC']

    df['FirstFVC'] =  df['Patient'].map(first_FVC_dict)

    

    return df
def calculate_height(row):

    if row['Sex'] == 'Male':

        return row['FirstFVC'] / (27.63 - 0.112 * row['Age'])

    else:

        return row['FirstFVC'] / (21.78 - 0.101 * row['Age'])

    
df = get_weeks_passed(df)

df = get_baseline_FVC(df)

df['Height'] = df.apply(calculate_height, axis=1)
df.head()
# import the necessary Encoders & Transformers

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.compose import ColumnTransformer



# define which attributes shall not be transformed, are numeric or categorical

no_transform_attribs = ['Patient', 'Weeks', 'MinWeek','mid_ct_scan','FVC']

num_attribs = ['Percent', 'Age', 'WeeksPassed', 'FirstFVC','Height']

cat_attribs = ['Sex', 'SmokingStatus']
from sklearn.base import BaseEstimator, TransformerMixin



class NoTransformer(BaseEstimator, TransformerMixin):

    """Passes through data without any change and is compatible with ColumnTransformer class"""

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X
## transform features into series



# create an instance of the ColumnTransformer

datawrangler = ColumnTransformer(([

     # the No-Transformer does not change the data and is applied to all no_transform_attribs 

     ('original', NoTransformer(), no_transform_attribs),

     # Apply StdScaler to the numerical attributes, here you can change to e.g. MinMaxScaler()   

     ('StdScaler', StandardScaler(), num_attribs),

     # OneHotEncoder all categorical attributes.   

     ('cat_encoder', OneHotEncoder(), cat_attribs),

    ]))



transformed_data_series = []

transformed_data_series = datawrangler.fit_transform(df)
## put transformed series into dataframe



# get column names for non-categorical data

new_col_names = no_transform_attribs + num_attribs



# extract possible values from the fitted transformer

categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]

new_col_names += categorical_values



# create Dataframe based on the extracted Column-Names

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)

train_sklearn_df.head()
csv_features_list = ['Percent','Age','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']

ctscan_features_list = ['mid_ct_scan']



X = train_sklearn_df[csv_features_list].astype(float)

X['mid_ct_scan'] = train_sklearn_df[ctscan_features_list]



y = train_sklearn_df[['FVC']].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



X_train_img = X_train[ctscan_features_list]

# X_train_img = X_train_img['mid_ct_scan']

X_train_csv = X_train[csv_features_list]



X_test_img = X_test[ctscan_features_list]

# X_test_img = X_test_img['mid_ct_scan']

X_test_csv = X_test[csv_features_list]
X_train_img = X_train_img['mid_ct_scan'].to_numpy()

X_train_img = np.stack( X_train_img, axis=0 )





X_test_img = X_test_img['mid_ct_scan'].to_numpy()

X_test_img = np.stack( X_test_img, axis=0 )
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import concatenate

from tensorflow.keras.utils import plot_model
# to-do pre-trained ResNet / VGG
def createDualInputModel():

    

    # Left for image

    Lin = Input(shape=(224,224,1), name = 'ctscan')

    Lx = Conv2D(32,(3,3),padding='same',activation='relu')(Lin)

    Lx = MaxPooling2D(pool_size=(2,2))(Lx)

    Lx = Flatten()(Lx)

    Lx = Dense(128,activation='relu')(Lx)

    

    # Right for csv

    Rin = Input((len(csv_features_list),), name = "csv")

    Rx = Dense(128,activation='relu')(Rin)

    

    # concatenate

    x = concatenate([Lx,Rx],axis=-1)

    x = Dense(128,activation='relu')(x)

    x = Dense(64,activation='relu')(x)

    x = Dense(1, activation='linear')(x) # no activation function since regression problem

    

    model = Model(inputs=[Lin,Rin],outputs=x)

    model.compile(loss='mean_absolute_error',

                  optimizer='rmsprop',

                  metrics=['mean_absolute_error'])

    

    return model
nn_model = createDualInputModel()

nn_model.summary()
nn_model.fit([X_train_img,X_train_csv], y_train,

             validation_data=([X_test_img,X_test_csv], y_test),

             epochs=10,

             batch_size=32,

             shuffle=True) #, callbacks=callbacks_list
import random

import tensorflow as tf

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
def seed_everything(seed): 

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

seed_everything(0)
features_list = ['Percent','Age','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']



X = train_sklearn_df[features_list].astype(float)

y = train_sklearn_df[['FVC']].astype(float)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror',

                          colsample_bytree = 0.3,

                          learning_rate = 0.1,

                          max_depth = 5,

                          alpha = 10,

                          n_estimators = 10)

xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
mae = mean_absolute_error(y_test, preds)

mae
base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

df = pd.read_csv(base_path + 'test.csv')

df = df.rename(columns={'Weeks':'MinWeek'})



base_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/'

df_submission = pd.read_csv(base_path + 'sample_submission.csv')

df_submission



# merge predictions from test set into submission set

df_submission[['Patient','Weeks']] = df_submission['Patient_Week'].str.split("_",expand=True,)

df_submission['Weeks'] = df_submission['Weeks'].astype('int')

# df = df.drop(['Weeks'], axis=1)

df_submission = df_submission.drop(['FVC','Confidence'],axis=1)

df_submission = pd.merge(df_submission, df, on=['Patient'], how='left')

df_submission
# Create base director for test .dcm files

director = "../input/osic-pulmonary-fibrosis-progression/test"



# Create path column with the path to each patient's CT

df_submission["Path"] = director + "/" + df_submission["Patient"]



# Create variable that shows how many CT scans each patient has

df_submission["CT_number"] = 0



for k, path in enumerate(df_submission["Path"]):

    df_submission["CT_number"][k] = len(os.listdir(path))
df_submission['mid_ct_scan'] = df_submission['Path'].apply(lambda x: get_mid_ct_scan(x))
# Feature Engineering

df_submission['WeeksPassed'] = df_submission['Weeks'] - df_submission['MinWeek']

df_submission = get_baseline_FVC(df_submission)

df_submission['Height'] = df_submission.apply(calculate_height, axis=1)

df_submission1 = df_submission[['Patient','Weeks','FVC','Percent','Age','Sex','SmokingStatus','Path','CT_number','mid_ct_scan','MinWeek','WeeksPassed','FirstFVC','Height']]
transformed_data_series = datawrangler.transform(df_submission1)
df_submission1['Patient_Week'] = df_submission['Patient_Week']
df_submssions = df_submission1
## put transformed series into dataframe



# get column names for non-categorical data

new_col_names = no_transform_attribs + num_attribs



# extract possible values from the fitted transformer

categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]

new_col_names += categorical_values



# create Dataframe based on the extracted Column-Names

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)

train_sklearn_df.head(6)
csv_features_list = ['Percent','Age','WeeksPassed','FirstFVC','Height','x0_Female','x1_Currently smokes','x1_Ex-smoker']

ctscan_features_list = ['mid_ct_scan']



X = train_sklearn_df[csv_features_list].astype(float)

X['mid_ct_scan'] = train_sklearn_df[ctscan_features_list]



# y = train_sklearn_df[['FVC']].astype(float)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)



X_train_img = X[ctscan_features_list]

# X_train_img = X_train_img['mid_ct_scan']

X_train_csv = X[csv_features_list]
X_train_img = X_train_img['mid_ct_scan'].to_numpy()

X_train_img = np.stack( X_train_img, axis=0 )
preds = nn_model.predict([X_train_img,X_train_csv],

             batch_size=32) #, callbacks=callbacks_list
df_submission['FVC'] = preds

df_submission['Confidence'] = 100
df_submission = df_submission[['Patient_Week','FVC','Confidence']]

df_submission.to_csv('/kaggle/working/submission.csv', index=False)

# def create_cnn_model():

#     Lin = Input(shape=(224,224,1), name = 'ctscan')

#     Lx = Conv2D(1, (3,3),padding='same',activation='relu')(Lin)

#     Lx = MaxPooling2D(pool_size=(2,2))(Lx)

#     Lx = Flatten()(Lx)

#     Lx = Dense(64)(Lx)

#     Lx = Dense(1)(Lx)



    

#     model = Model(inputs=Lin,outputs=Lx)

#     model.compile(loss='categorical_crossentropy',

#                   optimizer='rmsprop',

#                   metrics=['accuracy'])

    

#     return model
# cnn_model = create_cnn_model()

# cnn_model.summary()

# cnn_model.fit(X_train_img,y_train,

#               validation_data=(X_test_img, y_test),

#               epochs=10,

#               batch_size=16,

#               shuffle=True)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import scipy as sp

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from typing import Dict

import pydicom

import glob, os, tqdm

import warnings

from sklearn.metrics import mean_squared_error



warnings.filterwarnings('ignore')
os.getcwd()
meta = pd.read_csv('/kaggle/input/metadatapf/meta_data.csv') #meta data from CT images

train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
#creating single entry of features from CT data across all dicoms from single session

df_meta = meta.groupby(['Patient']).agg(

    {

     'img_mean': ['mean', 'std'],

     'img_std':['mean', 'std']

    }

)

df_meta.columns = df_meta.columns.map('_'.join)

df_meta = df_meta.reset_index()
#3 patients in the training set did not have DCM images, thus they are removed and only those on whom we have CT are kept (should consider a scenario when dicom data is not available in test?)

df_patient = pd.merge(left=df_meta, right=train, how='left', on='Patient')

#df_patient.head()
#Doing one_hot_encoding on cateogrical variables, Sex and Smoking Status



sex_dummies = pd.get_dummies(df_patient.Sex)

smoking_dummies = pd.get_dummies(df_patient.SmokingStatus)

df = pd.concat([df_patient, sex_dummies, smoking_dummies], axis=1)

df.drop(columns=['Sex', 'SmokingStatus'], inplace=True)

#df.head() #df now has training data of all patients on whom we also have dicoms available
# Since the original training data contained patients which are also in the test set, we decided to remove those patients from training set

train_patient_ids = set(df['Patient'].unique())

test_patient_ids = set(test['Patient'].unique())



train_no_test_ids = train_patient_ids.intersection(test_patient_ids) #just identifying those patients from test set which are in training set as well



if train_no_test_ids: #removing test data if there is some overlap, else not

    for id in train_no_test_ids:

        df = df.loc[df.Patient != id]

        

#df_test = df.copy()

#df_test_empty = pd.DataFrame(columns = df.columns)



# final train dataset "df"
X = df.drop(columns=['FVC', 'Patient', 'Percent']) #dropping patient ID and FVC (as that is to be predicted) and Percentage (as that is dependent on FVC so can't use)

y = df['FVC'] # separating out the predicted feature
X.head()
#creating a very simple linear regression model with all the features and all data from all test patients

lm = LinearRegression()

model = lm.fit(X,y)
model.coef_
def extract_dicom_meta_data(filename: str) -> Dict:

    # Load image

    

    image_data = pydicom.read_file(filename)

    img=np.array(image_data.pixel_array).flatten()

    row = {

        'Patient': image_data.PatientID,

        'body_part_examined': image_data.BodyPartExamined,

        'image_position_patient': image_data.ImagePositionPatient,

        'image_orientation_patient': image_data.ImageOrientationPatient,

        'photometric_interpretation': image_data.PhotometricInterpretation,

        'rows': image_data.Rows,

        'columns': image_data.Columns,

        'pixel_spacing': image_data.PixelSpacing,

        'window_center': image_data.WindowCenter,

        'window_width': image_data.WindowWidth,

        'modality': image_data.Modality,

        'StudyInstanceUID': image_data.StudyInstanceUID,

        'SeriesInstanceUID': image_data.StudyInstanceUID,

        'StudyID': image_data.StudyInstanceUID, 

        'SamplesPerPixel': image_data.SamplesPerPixel,

        'BitsAllocated': image_data.BitsAllocated,

        'BitsStored': image_data.BitsStored,

        'HighBit': image_data.HighBit,

        'PixelRepresentation': image_data.PixelRepresentation,

        'RescaleIntercept': image_data.RescaleIntercept,

        'RescaleSlope': image_data.RescaleSlope,

        'img_min': np.min(img),

        'img_max': np.max(img),

        'img_mean': np.mean(img),

        'img_std': np.std(img)}



    return row
#extracting image data from test CTs (this step can take some time)

test_image_path = '/kaggle/input/osic-pulmonary-fibrosis-progression/test'

test_image_files = glob.glob(os.path.join(test_image_path, '*', '*.dcm'))



meta_data_test = []

for filename in tqdm.tqdm(test_image_files):

    try:

        meta_data_test.append(extract_dicom_meta_data(filename))

    except Exception as e:

        continue
meta_data_test = pd.DataFrame.from_dict(meta_data_test) #make meta data from test as a pd dataframe

meta_data_test.head()
meta_data_test.shape
#creating single entry of features from CT data of test patients across all dicoms from single session

df_meta_test = meta_data_test.groupby(['Patient']).agg(

    {

     'img_mean': ['mean', 'std'],

     'img_std':['mean', 'std']

    }

)

df_meta_test.columns = df_meta_test.columns.map('_'.join)

df_meta_test = df_meta_test.reset_index()
df_patient_test = pd.merge(left=df_meta_test, right=test, how='left', on='Patient')

df_patient_test.head()
#we first need to make sure that all levels of all categories are covered in test data

df_patient_test['Sex'] = pd.Categorical(df_patient_test['Sex'], categories=['Male', 'Female'])

df_patient_test['SmokingStatus'] = pd.Categorical(df_patient_test['SmokingStatus'], categories=['Ex-smoker', 'Never smoked', 'Currently smokes'])



sex_dummies = pd.get_dummies(df_patient_test.Sex)

smoking_dummies = pd.get_dummies(df_patient_test.SmokingStatus)

df_test = pd.concat([df_patient_test, sex_dummies, smoking_dummies], axis=1)

df_test.drop(columns=['Sex', 'SmokingStatus'], inplace=True)

df_test.head() #df_test now has testing data of all patients in test folder (what if)
df_test_final = df_test.drop(columns=['FVC', 'Patient', 'Percent'])

y_test = df_test['FVC']
X.head()
df_test_final
y_pred = lm.predict(df_test_final)
y_pred
y_test
rmse_pred = mean_squared_error(y_test, y_pred)
rmse_pred
def metric(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

        Calculates the modified Laplace Log Likelihood score for this competition.

        Credits: https://www.kaggle.com/rohanrao/osic-understanding-laplace-log-likelihood

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)
score = metric(y_test, y_pred, np.std(y_pred))



print('OOF log-Laplace likelihood score:', score)
#still to work on it.....

'''

otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i] #all they are doing is still using the test data while training, and also predicting on it just that replacing their prediction with the real value in test data

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1



subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)

'''
df_test
df_test_sub = pd.DataFrame(columns=['Patient', 'Weeks'])

for id in df_test.Patient:

    data = {'Patient': [id for x in range(-12, 134)],

                  'Weeks': [x for x in range(-12, 134)]

                  }

    df_inter = pd.DataFrame(data, columns=['Patient', 'Weeks'])

    df_test_sub = pd.concat([df_test_sub, df_inter])
df_test_sub
df_test_sub_final = pd.merge(left=df_test, right=df_test_sub, on='Patient', how='right')

df_test_sub_final.drop(columns=['Weeks_x'], inplace=True)

df_test_sub_final.rename(columns={'Weeks_y': 'Weeks'}, inplace=True)

df_test_sub_final
df_test_rm = df_test_sub_final.drop(columns=['Patient', 'FVC', 'Percent'])

df_test_sub_final['FVC_pred'] = lm.predict(df_test_rm)

df_test_sub_final
final_submission = df_test_sub_final[['Patient', 'Weeks']]

final_submission['Patient_Week'] = final_submission['Patient'] + "_" + final_submission['Weeks'].astype(str)

final_submission['FVC'] = df_test_sub_final[['FVC_pred']]

final_submission = final_submission.drop(columns=['Patient', 'Weeks'])

final_submission['Confidence'] = 100

final_submission
final_submission.to_csv("submission.csv", index=False)
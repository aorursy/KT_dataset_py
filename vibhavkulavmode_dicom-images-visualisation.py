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
IS_LOCAL = False

import numpy as np

import pandas as pd

from skimage.io import imread

import seaborn as sns

import matplotlib.pyplot as plt

from glob import glob

import pydicom as dicom

import os
PATH="../input/siim-medical-images/"

os.listdir(PATH)
data_df = pd.read_csv(os.path.join(PATH,"overview.csv"))
data_df.info()
data_df.head()
tiff_data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH+'tiff_images/*.tif')])

def process_data(path):

    data = pd.DataFrame([{'path': filepath} for filepath in glob(PATH+path)])

    data['file'] = data['path'].map(os.path.basename)

    data['ID'] = data['file'].map(lambda x: str(x.split('_')[1]))

    data['Age'] = data['file'].map(lambda x: int(x.split('_')[3]))

    data['Contrast'] = data['file'].map(lambda x: bool(int(x.split('_')[5])))

    data['Modality'] = data['file'].map(lambda x: str(x.split('_')[6].split('.')[-2]))

    return data
tiff_data = process_data('tiff_images/*.tif')
tiff_data.head(10)
dicom_data = process_data('dicom_dir/*.dcm')
dicom_data.head(10)
def show_images(data, dim=16, imtype='TIFF'):

    img_data = list(data[:dim].T.to_dict().values())

    f, ax = plt.subplots(4,4, figsize=(16,20))

    for i,data_row in enumerate(img_data):

        if(imtype=='TIFF'): 

            data_row_img = imread(data_row['path'])

        elif(imtype=='DICOM'):

            data_row_img = dicom.read_file(data_row['path'])

        if(imtype=='TIFF'):

            ax[i//4, i%4].matshow(data_row_img,cmap='gray')

        elif(imtype=='DICOM'):

            ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//4, i%4].axis('off')

        ax[i//4, i%4].set_title('Modality: {Modality} Age: {Age}\nSlice: {ID} Contrast: {Contrast}'.format(**data_row))

    plt.show()
show_images(tiff_data,16,'TIFF')
show_images(dicom_data,16,'DICOM')
dicom_file_path = list(dicom_data[:1].T.to_dict().values())[0]['path']

dicom_file_dataset = dicom.read_file(dicom_file_path)

dicom_file_dataset
print("Modality: {}\nManufacturer: {}\nPatient Age: {}\nPatient Sex: {}\nPatient Name: {}\nPatient ID: {}".format(

    dicom_file_dataset.Modality, 

    dicom_file_dataset.Manufacturer,

    dicom_file_dataset.PatientAge,

    dicom_file_dataset.PatientSex,

    dicom_file_dataset.PatientName,

    dicom_file_dataset.PatientID))
def show_dicom_images(data):

    img_data = list(data[:16].T.to_dict().values())

    f, ax = plt.subplots(4,4, figsize=(16,20))

    for i,data_row in enumerate(img_data):



        data_row_img = dicom.read_file(data_row['path'])

        modality = data_row_img.Modality

        age = data_row_img.PatientAge

        

        ax[i//4, i%4].imshow(data_row_img.pixel_array, cmap=plt.cm.bone) 

        ax[i//4, i%4].axis('off')

        ax[i//4, i%4].set_title('Modality: {} Age: {}\nSlice: {} Contrast: {}'.format(

         modality, age, data_row['ID'], data_row['Contrast']))

    plt.show()
show_dicom_images(dicom_data)
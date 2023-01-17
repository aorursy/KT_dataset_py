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

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import pydicom

from pydicom.data import get_testdata_files

import glob as glob



path = "../input/osic-pulmonary-fibrosis-progression/train/"

path1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/19.dcm"

path_patient1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/"

img1 = pydicom.dcmread(path1)



print(img1.pixel_array.shape)

plt.figure(figsize = (7, 7))

plt.imshow(img1.pixel_array, cmap="plasma")

plt.axis('off');
print("Patient id.......:", img1.PatientID, "\n" +

      "Modality.........:", img1.Modality, "\n" +

      "Rows.............:", img1.Rows, "\n" +

      "Columns..........:", img1.Columns)



print("img1:", img1)
data_path = '../input/osic-pulmonary-fibrosis-progression/train/'



output_path = '../input/output/'

train_image_files = sorted(glob.glob(os.path.join(data_path, '*','*.dcm')))

patients = os.listdir(data_path)

patients.sort()



print('Some sample Patient ID''s :', len(train_image_files))

print("\n".join(train_image_files[:5]))

def load_scan(path):

    slices = [pydicom.read_file(s) for s in path[0:10]]

    return slices
data_10_photos = load_scan(train_image_files)

img_data = [img.pixel_array for img in data_10_photos]

print(img_data)

print(img_data[0][100]) # the 100th raw in the first img
ids = [img.PatientID for img in data_10_photos]

ids
# import tabular data



tabular_dataset_train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

tabular_dataset_train.head()
# create new df with patient_id and mean_fvc 



new_df = tabular_dataset_train.groupby('Patient').mean('FVC').drop(columns = ['Age', 'Percent', 'Weeks'])

print(new_df)
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D





img_rows, img_cols = 512, 512

num_images = 10



x = np.array([img_data]).reshape(num_images, img_rows, img_cols, 1)

y = np.array([2113 for i in range(10)]) # mean fvc of that patient

    

model = Sequential()

model.add(Conv2D(20, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(1, activation='linear'))



model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])



model.summary()



history = model.fit(x, y,

          batch_size=1,

          epochs=30,

          validation_split = 0.2)
mae_train = history.history['mean_absolute_error']

mae_val = history.history['val_mean_absolute_error']

epochs = range(1,31)

plt.plot(epochs, mae_train, 'g', label='Training mae')

plt.plot(epochs, mae_val, 'b', label='validation mae')

plt.title('Training and Validation Mean Absolure Error')

plt.xlabel('Epochs')

plt.ylabel('MAE')

plt.legend()

plt.show()
mae_train = history.history['mean_absolute_error']

mae_val = history.history['val_mean_absolute_error']

epochs = range(1,31)

plt.plot(epochs, mae_train, 'g', label='Training mae')

plt.plot(epochs, mae_val, 'b', label='validation mae')

plt.title('Training and Validation Mean Absolure Error (log scale)')

plt.xlabel('Epochs')

plt.ylabel('MAE')

plt.yscale("log")

plt.legend()

plt.show()
def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    return slices
# import all CT scans



from pathlib import Path

root_dir = Path('/kaggle/input/osic-pulmonary-fibrosis-progression/train')

def load_scan(path):

    slices = [pydicom.read_file(p) for p in path.glob('*.dcm')]

    image = np.stack([s.pixel_array.astype(float) for s in slices])

    return image, slices[0]
# type(slices[0])
# add FVC values to scans data
# create CNN model with: X = scans, y = FVC
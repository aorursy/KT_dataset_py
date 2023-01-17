import matplotlib.pyplot as plt

import pydicom

import json

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"

ds = pydicom.dcmread(filename)

plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"

pydicom.dcmread(filename).pixel_array.shape
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00136637202224951350618/353.dcm"

pydicom.dcmread(filename).pixel_array.shape
filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"

pydicom.dcmread(filename)
dir(pydicom.dcmread(filename))
dir(pydicom.dcmread(filename)['ImageOrientationPatient'])
pydicom.dcmread(filename)['ImageOrientationPatient'].to_json()
json.loads(pydicom.dcmread(filename)['ImageOrientationPatient'].to_json())['Value']
# directory for a patient

imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140"

print("total images for patient ID00123637202217151272140: ", len(os.listdir(imdir)))
print("images for patient ID00123637202217151272140 in a rough order:")

mylist = os.listdir(imdir)

mylist.sort()

print(mylist)
# view first (columns*rows) images in order

w=10

h=10

fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show()
files = folders = 0



path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)



print("{:,} files/images, {:,} folders/patients".format(files, folders))
files = []

for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files.append(len(filenames))



print("{:,} average files/images per patient".format(round(np.mean(files))))

print("{:,} max files/images per patient".format(round(np.max(files))))

print("{:,} min files/images per patient".format(round(np.min(files))))
files = folders = 0



path = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"



for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files += len(filenames)

    folders += len(dirnames)



print("{:,} files/images, {:,} folders/patients".format(files, folders))
files = []

for _, dirnames, filenames in os.walk(path):

  # ^ this idiom means "we won't be using this value"

    files.append(len(filenames))



print("{:,} average files/images per patient".format(round(np.mean(files))))

print("{:,} max files/images per patient".format(round(np.max(files))))

print("{:,} min files/images per patient".format(round(np.min(files))))
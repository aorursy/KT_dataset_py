# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

import glob

import os.path

import pandas as pd

import pydicom

import matplotlib.pyplot as plt

import seaborn as sb

import numpy as np

import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')



from mask_functions import rle2mask



from skimage import exposure
def create_input_dataset(directory):

    input_df = pd.DataFrame(glob.glob(directory + "/**/*.dcm", recursive=True), columns = ["Path"])

    input_df["ImageId"] = input_df["Path"].apply(lambda x: os.path.basename(x)[:-4])

    def df_extract_metadata_from_path(df):

        dcm = pydicom.dcmread(df['Path'])

        df['Age'] = dcm.PatientAge

        df['Sex'] = dcm.PatientSex 

        df['Rows'] = dcm.Rows 

        df['Columns'] = dcm.Columns

        return df

    input_df = input_df.apply(df_extract_metadata_from_path, axis=1)

    input_df['Age'] = input_df['Age'].astype('int32')

    return input_df
def create_label_dataset(filename, width=1024, height=1024):

    NO_MASK_STRING = ' -1'

    labels_df = pd.read_csv(filename)

    # Fix typo in target file

    labels_df = labels_df.rename({" EncodedPixels":"EncodedPixels"}, axis=1)

    # Images may contain multiple lre labels --> combine in df

    labels_df = pd.DataFrame(labels_df.groupby("ImageId")["EncodedPixels"].apply(list))

    labels_df['HasMask'] = labels_df["EncodedPixels"].apply(lambda x: x != [NO_MASK_STRING])

    labels_df['NMasks'] = labels_df['EncodedPixels'].apply(lambda x: len(x) if x != [NO_MASK_STRING] else 0)

    labels_df['OverlappingMasks'] = labels_df['EncodedPixels'].apply(lambda x: (sum([rle2mask(i, width=width, height=height)/255 for i in x]) > 1).any() if len(x) > 1 else False)

    mask_coverages_df = labels_df[labels_df['HasMask']]['EncodedPixels'].apply(lambda y: list(map(lambda x: (rle2mask(x, width=width, height=height)/255).sum()/(width*height),y)))

    labels_df["MaskCoverage"] = mask_coverages_df.apply(sum)

    return labels_df
input_df = create_input_dataset("../input/siim-train-test/siim/dicom-images-train")

test_df = create_input_dataset("../input/siim-train-test/siim/dicom-images-test")

labels_df = create_label_dataset("../input/siim-train-test/siim/train-rle.csv")
def check_input_dataset(df):

    rows = df['Rows'].unique()

    columns = df['Columns'].unique()

    if len(rows) != 1 or len(columns) != 1:

        raise RuntimeError("ERROR: input images don't have the same size")

    print("Image dimensions: %d x %d" % (rows[0], columns[0]))

    print("Input data shape: %s" % str(df.shape))
def merge_input_with_labels(input_df, labels_df):

    print(10*"*" + " Before merge " + 10*"*")

    print("Input Data Shape:  %s" % str(input_df.shape)) 

    print("Labels Shape:      %s" % str(labels_df.shape))

    df = pd.merge(labels_df, input_df, on="ImageId", validate="one_to_one")

    print(10*"*" + " After merge " + 10*"*")

    print("Merged data shape: %s" % str(df.shape))

    return df
check_input_dataset(input_df)
train_df = merge_input_with_labels(input_df, labels_df)
train_df.groupby("HasMask")["ImageId"].count()
print(train_df["NMasks"].describe())

print((train_df[train_df["HasMask"]]["NMasks"] >= 2).sum())

sb.distplot(train_df["NMasks"], kde=False)
overlaps = train_df[train_df["NMasks"] >= 2]

overlaps[overlaps["OverlappingMasks"]]["ImageId"]

overlaps.to_csv("overlaps_IDs.csv")
ax = sb.distplot(train_df['Age'])

ax = sb.distplot(test_df['Age'])

ax.set_xlim(0,100)
train_df.groupby("Sex").count()["ImageId"]/len(train_df)
test_df.groupby("Sex").count()["ImageId"]/len(test_df)
ax = sb.distplot(train_df[train_df["HasMask"]]['Age'])

ax = sb.distplot(train_df[~train_df["HasMask"]]['Age'])

ax.set_xlim(0,100)
ax = sb.distplot(train_df[(train_df["HasMask"]) & (train_df["Sex"] == "M")]['Age'], label="Mask && M")

ax = sb.distplot(train_df[(~train_df["HasMask"]) & (train_df["Sex"] == "M")]['Age'], label="NoMask && M")

plt.legend()

ax.set_xlim(0,100)
ax = sb.distplot(train_df[(train_df["HasMask"]) & (train_df["Sex"] == "F")]['Age'], label="Mask && F")

ax = sb.distplot(train_df[(~train_df["HasMask"]) & (train_df["Sex"] == "F")]['Age'], label="NoMask && F")

plt.legend()

ax.set_xlim(0,100)
fig, ax = plt.subplots()

sb.distplot(train_df[~train_df['MaskCoverage'].isnull()]['MaskCoverage'])
sb.scatterplot(train_df["Age"], train_df[~train_df['MaskCoverage'].isnull()]['MaskCoverage'])
def equalize_image(img):

    cdf, centroids = exposure.cumulative_distribution(img)

    return np.interp(img, centroids, cdf)
def show_image_and_masks(index, maskindex=-1):

    fig, ax = plt.subplots(1, 3, figsize=(25,45))

    dcm = pydicom.dcmread(train_df["Path"][index])

    image = dcm.pixel_array/255

    image_equalized = equalize_image(dcm.pixel_array/255)

    ax[0].imshow(image_equalized, cmap='bone')

    ax[1].imshow(image_equalized, cmap='bone')



    if train_df['HasMask'][index]:

        print("Image has %d masks" % len(train_df['EncodedPixels'][index]))

        if maskindex == -1:

            mask = sum([rle2mask(x, width=1024, height=1024).T for x in train_df['EncodedPixels'][index]])

        else:

            mask = rle2mask(train_df['EncodedPixels'][index][maskindex], width=1024, height=1024).T

        ax[1].imshow(mask, cmap='Reds', alpha=0.3, interpolation='none')

        ax[2].imshow(mask, cmap='Reds', interpolation='none')





    plt.show()
train_df[train_df["NMasks"] == 10]['Path']
show_image_and_masks(13)
show_image_and_masks(5415)
train_df["ImageId"][5415]
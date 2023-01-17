# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pydicom

import os
pd.options.display.max_rows = 500

pd.options.display.max_columns = 100

pd.options.display.max_colwidth = 200
TRAIN_IMG_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/"

TEST_IMG_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/"

TRAIN_DATA_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"

SUBMISSION_PATH = "/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"
def loaddataset(input):

    """

    Read csv file and return dataframe

    """

    return pd.read_csv(input)



def removeduplicates(df, _keep='first', _inplace=True):

    df_copy = df.copy()

    """

    Remove duplicates keeping first row.

    """

    df_copy = df_copy.drop_duplicates(keep=_keep, inplace=_inplace)

    return df_copy



def splitcolumn(df):

    """

    Method read column value and split in multiple columns. This is very much specific to data.

    """

    df['Hemorrhage'] = df['ID'].apply(lambda x : str(x).rsplit('_',1)[1])

    df['ID'] = df['ID'].apply(lambda x : str(x).rsplit('_',1)[0])

    return df



def pivot_dataframe(args, df, column_name, value, indexcolumn, IMG_PATH):

    """

    This method is used to convert row wise data to column wise.

    column name for which pivoting is required.

    value to keep for pivoted columns

    indexcolumn on which dataframe has to be indexed.

    """

    df = pd.pivot_table(df, columns=column_name, values=value, index=indexcolumn).reset_index()

    df['filepath'] = f'{IMG_PATH}' + df['ID'] + f'.dcm'

    return df
train = loaddataset(TRAIN_DATA_PATH)

test = loaddataset(SUBMISSION_PATH)

print(f'Shape of train dataset before removing duplicates {train.shape}')

print(f'Shape of test dataset before removing duplicates {test.shape}')

removeduplicates(train, 'first', True)

removeduplicates(test, 'first', True)

print(f'Shape of train dataset after removing duplicates {train.shape}')

print(f'Shape of test dataset after removing duplicates {test.shape}')

train = splitcolumn(train)

test = splitcolumn(test)

train = pivot_dataframe(train, train,['Hemorrhage'],'Label',['ID'], TRAIN_IMG_PATH)

test = pivot_dataframe(test, test,['Hemorrhage'],'Label',['ID'], TEST_IMG_PATH)
train.head()
test.head()
def get_dicom_value(x, cast=int):

    if type(x) in [pydicom.multival.MultiValue, tuple]:

        return cast(x[0])

    else:

        return cast(x)





def cast(value):

    if type(value) is pydicom.valuerep.MultiValue:

        return tuple(value)

    return value





def get_dicom_raw(dicom):

    return {attr:cast(getattr(dicom,attr)) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']}





def rescale_image(image, slope, intercept, bits, pixel):

    # In some cases intercept value is wrong and can be fixed

    # Ref. https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai

    if bits == 12 and pixel == 0 and intercept > -100:

        image = image.copy() + 1000

        px_mode = 4096

        image[image>=px_mode] = image[image>=px_mode] - px_mode

        intercept = -1000

    return image.astype(np.float32) * slope + intercept





def apply_window(image, center, width):

    image = image.copy()

    min_value = center - width // 2

    max_value = center + width // 2

    image[image < min_value] = min_value

    image[image > max_value] = max_value

    return image





def get_windowed_ratio(image, center, width):

    # get ratio of pixels within the window

    windowed = apply_window(image, center, width)

    return len(np.where((windowed > 0) & (windowed < 80))[0]) / windowed.size
def create_record(id, img_path):



    #id, labels = item



    #path = '%s/%s.dcm' % (dirname, id)

    dicom = pydicom.dcmread(img_path)

    

    record = {

        'ID': id,

    }

    record.update(get_dicom_raw(dicom))

    raw = dicom.pixel_array

    slope = float(record['RescaleSlope'])

    intercept = float(record['RescaleIntercept'])

    center = get_dicom_value(record['WindowCenter'])

    width = get_dicom_value(record['WindowWidth'])

    bits= record['BitsStored']

    pixel = record['PixelRepresentation']



    image = rescale_image(raw, slope, intercept, bits, pixel)

    doctor = apply_window(image, center, width)

    brain = apply_window(image, 40, 80)



    record.update({

        'raw_max': raw.max(),

        'raw_min': raw.min(),

        'raw_mean': raw.mean(),

        'raw_std' : raw.std(),

        'raw_diff': raw.max() - raw.min(),

        'doctor_max': doctor.max(),

        'doctor_min': doctor.min(),

        'doctor_mean': doctor.mean(),

        'doctor_std' : doctor.std(),

        'doctor_diff': doctor.max() - doctor.min(),

        'brain_max': brain.max(),

        'brain_min': brain.min(),

        'brain_mean': brain.mean(),

        'brain_std' : brain.std(),

        'brain_diff': brain.max() - brain.min(),

        'brain_ratio': get_windowed_ratio(image, 40, 80),

    })

    return record
def dicommetadata(df):

    dicom_metadata=[]

    ids = []

    for index, row in df.iterrows():

        try:

            record = create_record(row['ID'], row['filepath'])

            dicom_metadata.append(record)

        except:

            ids.append(row['ID'])

            continue

    return dicom_metadata, ids
dicom_metadata, corrupted_ids = dicommetadata(train)

dicom_metadata_test, corrupted_ids_test = dicommetadata(test)

dicom_df = pd.DataFrame(dicom_metadata)

dicom_df_test = pd.DataFrame(dicom_metadata_test)

dicom_df.to_pickle('dicom_df.pkl')

dicom_df_test.to_pickle('dicom_df_test.pkl')
print("train corrupted ids ", len(corrupted_ids))

print("test corrupted ids ", len(corrupted_ids_test))

print(dicom_df.shape)

print(dicom_df_test.shape)
def remove_corrupted_images(ids, df):

    df_temp = df.copy()

    for id in ids:

        df_temp = df_temp.drop(df_temp[df_temp['ID'] == id].index, axis=0)

    return df_temp
train_uncorrupted = remove_corrupted_images(corrupted_ids, train)

test_uncorrupted = remove_corrupted_images(corrupted_ids_test, test)

train_uncorrupted.to_pickle('train_uncorrupted.pkl')

test_uncorrupted.to_pickle('test_uncorrupted.pkl')
print("train uncorrupted shape ", train_uncorrupted.shape)

print("test uncorrupted shape ", test_uncorrupted.shape)
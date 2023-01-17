import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import pydicom

import re

import os

from pathlib import Path



plt.style.use('grayscale')

ROOT_DIR = Path('../input/rsna-intracranial-hemorrhage-detection')
! ls {ROOT_DIR}
TRAIN_DIR = ROOT_DIR/'stage_1_train_images'

TEST_DIR = ROOT_DIR/'stage_1_test_images'
! ls {TRAIN_DIR} | wc -l
! ls {TRAIN_DIR} | head -n 5
train_df = pd.read_csv(ROOT_DIR/'stage_1_train.csv')

print(train_df.shape)

train_df.head(10)

train_df[['ID', 'Subtype']] = train_df['ID'].str.rsplit(pat='_', n=1, expand=True)

print(train_df.shape)

train_df.head()

def fix_id(img_id, img_dir=TRAIN_DIR):

    if not re.match(r'ID_[a-z0-9]+', img_id):

        sop = re.search(r'[a-z0-9]+', img_id)

        if sop:

            img_id_new = f'ID_{sop[0]}'

            return img_id_new

        else:

            print(img_id)

    return img_id



# test

assert(fix_id('ID_63eb1e259') == fix_id('ID63eb1e259'))

test = 'ID_dbdedfada'

assert(fix_id(test) == 'ID_dbdedfada')

train_df['ID'] = train_df['ID'].apply(fix_id)

# this method also handles duplicates gracefully

train_new = train_df.pivot_table(index='ID', columns='Subtype').reset_index()

print(train_new.shape)

train_new.head()

subtype_ct = train_new['Label'].sum(axis=0)

print(subtype_ct)

sns.barplot(x=subtype_ct.values, y=subtype_ct.index);
def id_to_filepath(img_id, img_dir=TRAIN_DIR):

    filepath = f'{img_dir}/{img_id}.dcm' # pydicom doesn't play nice with Path objects

    if os.path.exists(filepath):

        return filepath

    else:

        return 'DNE'

img_id = train_new['ID'][0]

img_filepath = id_to_filepath(img_id)

print(img_filepath)
train_new['filepath'] = train_new['ID'].apply(id_to_filepath)

train_new.head()

dcm_data = pydicom.dcmread(img_filepath)

print(dcm_data)

def get_patient_data(filepath):

    if filepath != 'DNE':

        dcm_data = pydicom.dcmread(filepath, stop_before_pixels=True)

        return dcm_data.PatientID, dcm_data.StudyInstanceUID, dcm_data.SeriesInstanceUID

patient, study, series = get_patient_data(img_filepath)

print(patient, study, series)

# quick test to make sure our df.apply syntax is working, since the next cell takes a long time to run

test = train_new[:5].copy()

test['PatientID'], test['StudyID'], test['SeriesID'] = zip(*test['filepath'].map(get_patient_data))

test.head()

train_new['PatientID'], train_new['StudyID'], train_new['SeriesID'] = zip(*train_new['filepath'].map(get_patient_data))

print(train_new.shape[0])

print(len(train_new['PatientID'].unique()))

print(len(train_new['StudyID'].unique()))

print(len(train_new['SeriesID'].unique()))

type(dcm_data.WindowWidth)
def window_img(dcm, width=None, level=None):

    pixels = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    if not width:

        width = dcm.WindowWidth

        if type(width) != pydicom.valuerep.DSfloat:

            width = width[0]

    if not level:

        level = dcm.WindowCenter

        if type(level) != pydicom.valuerep.DSfloat:

            level = level[0]

    lower = level - (width / 2)

    upper = level + (width / 2)

    return np.clip(pixels, lower, upper)



def load_one_image(idx, df=train_new, width=None, level=None):

    assert('filepath' in df.columns)

    dcm_data = pydicom.dcmread(df['filepath'][idx])

    pixels = window_img(dcm_data, width, level)

    return pixels

# standard brain window

pixels = load_one_image(0)

plt.imshow(pixels);

# subdural window

pixels_new = load_one_image(0, width=200, level=80)

plt.imshow(pixels_new);

def show_examples(subtype='epidural', df=train_new):

    df_new = df.set_index('ID')

    filt = df_new['Label'][subtype] == 1

    df_new = df_new[filt]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(16):

        idx = df_new.index[i]

        pixels = load_one_image(idx, df_new)

        a = i // 4

        b = i % 4

        axes[a, b].imshow(pixels)

show_examples('epidural')
show_examples('subdural')
show_examples('intraventricular')
show_examples('intraparenchymal')
show_examples('subarachnoid')
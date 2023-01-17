# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Plotting

import seaborn as sns # statistical data visualization

import plotly.express as px

import plotly.graph_objects as go # interactive plots



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

train_df.head()
print('Number of data points: ' + str(len(train_df)))

print('----------------------')



for col in train_df.columns:

    print('{} : {} unique values, {} missing.'.format(col, 

                                                          str(len(train_df[col].unique())), 

                                                          str(train_df[col].isna().sum())))
unique_patient_df = train_df.drop(['Weeks', 'FVC', 'Percent'], axis=1).drop_duplicates().reset_index(drop=True)

unique_patient_df['# visits'] = [train_df['Patient'].value_counts().loc[pid] for pid in unique_patient_df['Patient']]



print('Number of data points: ' + str(len(unique_patient_df)))

print('----------------------')



for col in unique_patient_df.columns:

    print('{} : {} unique values, {} missing.'.format(col, 

                                                          str(len(unique_patient_df[col].unique())), 

                                                          str(unique_patient_df[col].isna().sum())))

unique_patient_df.head()
plt.figure(figsize=(10,5))

sns.countplot(x='Age', data=unique_patient_df)
plt.figure(figsize=(5,3))

sns.countplot(x='Sex', data=unique_patient_df)
plt.figure(figsize=(10, 3))

sns.countplot(x='SmokingStatus', data=unique_patient_df)
plt.figure(figsize=(20, 5))

ax = sns.countplot(x='Weeks', data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation='vertical')



plt.figure(figsize=(5, 5))

sns.countplot(x='# visits', data=unique_patient_df)
plt.figure(figsize=(10, 5))

sns.distplot(train_df['FVC'], rug=True)
plt.figure(figsize=(10, 5))

sns.distplot(train_df['Percent'], rug=True)
train_df['Expected FVC'] = train_df['FVC'] * (100/train_df['Percent'])



plt.figure(figsize=(10, 5))

sns.distplot(train_df['Expected FVC'], rug=True)
train_df['FVC Difference'] = train_df['Expected FVC'] - train_df['FVC']



plt.figure(figsize=(10, 5))

sns.distplot(train_df['FVC Difference'], rug=True)
pd.crosstab(train_df.Sex, train_df.SmokingStatus, margins=True, normalize=True)
plt.figure(figsize=(10,10))

sns.heatmap(train_df.apply(lambda x : pd.factorize(x)[0] if x.dtype=='object' else x).corr(method='pearson'),

            annot=True)
fig = px.line(train_df, 'Weeks', 'FVC', color='Patient',

             title='Pulmonary Condition Progression by Patient',

             labels={'Week':'Week #',

                     'FVC' : 'Actual\Expected FVC'})

fig.update_traces(mode='lines+markers')



for i in fig.data:

    new_element = go.Scattergl(i)

    new_element['mode'] = 'markers'

    new_element['name'] = i['name'] + '_EX'

    new_element['y'] = train_df.loc[train_df['Patient'] == new_element['legendgroup']]['Expected FVC']

    fig.add_trace(new_element)





fig.show()
fig = px.line(train_df, 'Weeks', 'FVC', line_group='Patient', color='Sex',

             title='Pulmonary Condition Progression by Sex')

fig.update_traces(mode='lines+markers')
fig = px.line(train_df, 'Weeks', 'FVC', line_group='Patient', color='SmokingStatus',

             title='Pulmonary Condition Progression by Sex')

fig.update_traces(mode='lines+markers')
import pydicom

from glob import glob

import scipy.ndimage

from skimage import morphology

from skimage import measure

from skimage.filters import threshold_otsu, median

from scipy.ndimage import binary_fill_holes

from skimage.segmentation import clear_border

from scipy.stats import describe
patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/'

scans = glob(patient_dir + '/*.dcm')



scans[:5]
def load_scan(path):

    scans = os.listdir(path)

    slices = []

    

    for scan in scans:

        with pydicom.dcmread(path + '/' + scan) as s:

            slices.append(s)

    

    slices.sort(key = lambda x: int(x.InstanceNumber))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        try:

            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        except:

            slice_thickness = slices[0].SliceThickness

    

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



def get_pixels_hu(scans):

    image = np.stack([s.pixel_array for s in scans])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 1

    # The intercept is usually -1024, so air is approximately 0

    image[image <= -1000] = 0

    

    # Convert to Hounsfield units (HU)

    intercept = scans[0].RescaleIntercept

    slope = scans[0].RescaleSlope

    

    if slope != 1:

        image = slope * image.astype(np.float64)

        image = image.astype(np.int16)

        

    image += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
patient = load_scan(patient_dir)

imgs = get_pixels_hu(patient)



fig, ax = plt.subplots(1, 2, figsize=(10, 10))



ax[0].set_title('Original Image')

ax[0].imshow(patient[15].pixel_array, cmap='gray')

ax[0].axis('off')



ax[1].set_title('HU Image')

ax[1].imshow(imgs[15], cmap='gray')

ax[1].axis('off')
plt.hist(imgs.flatten(), bins=50, color='c')

plt.xlabel("Hounsfield Units (HU)")

plt.ylabel("Frequency")

plt.show()
def sample_stack(stack, rows=2, cols=3, start_with=0, show_every=5):

    fig,ax = plt.subplots(rows,cols,figsize=[10,10])

    for i in range(rows*cols):

        ind = start_with + i*show_every

        ax[i // (rows+1), int(i % cols)].set_title('slice %d' % ind)

        ax[i // (rows+1), int(i % cols)].imshow(stack[ind],cmap='gray')

        ax[i // (rows+1), int(i % cols)].axis('off')

    plt.tight_layout()

    plt.show()



sample_stack(imgs)
def lung_segment(img, display=False):

    thresh = threshold_otsu(img)

    binary = img <= thresh



    lungs = median(clear_border(binary))

    lungs = morphology.binary_closing(lungs, selem=morphology.disk(7))

    lungs = binary_fill_holes(lungs)



    final = lungs*img

    final[final == 0] = np.min(img)

    

    if display:

        fig, ax = plt.subplots(1, 4, figsize=(15, 15))



        ax[0].set_title('HU Image')

        ax[0].imshow(img, cmap='gray')

        ax[0].axis('off')



        ax[1].set_title('Thresholded Image')

        ax[1].imshow(binary, cmap='gray')

        ax[1].axis('off')



        ax[2].set_title('Lungs Mask')

        ax[2].imshow(lungs, cmap='gray')

        ax[2].axis('off')



        ax[3].set_title('Final Image')

        ax[3].imshow(final, cmap='gray')

        ax[3].axis('off')

    

    return final, lungs



def lung_segment_stack(imgs):

    

    masks = np.empty_like(imgs)

    segments = np.empty_like(imgs)

    

    for i, img in enumerate(imgs):

        seg, mask = lung_segment(img)

        segments[i,:,:] = seg

        masks[i,:,:] = mask

        

    return segments, masks
lung_segment(imgs[15], display=True)
segmented, masks = lung_segment_stack(imgs)

sample_stack(segmented)
def lung_volume(patient, masks):

    slice_thickness = patient[0].SliceThickness

    pixel_spacing = patient[0].PixelSpacing

    

    return np.round(np.sum(masks) * slice_thickness * pixel_spacing[0] * pixel_spacing[1], 3)
lung_vol = lung_volume(patient, masks)

print('Lung Volume is: ' + str(lung_vol) + ' mm^3 (' + str(lung_vol/1e6) + ' liters)')
def hist_analysis(segmented, display=False):

    values = segmented.flatten()

    values = values[values >= -1000]

    

    if display:

        plt.hist(values, bins=50)

    

    summary_statistics = describe(values)

    

    return summary_statistics
h_stat = hist_analysis(segmented, display=True)



print('Mean is: ' + str(h_stat.mean))

print('Variance is: ' + str(h_stat.variance))

print('Skewness is: ' + str(h_stat.skewness))

print('Kurtosis is: ' + str(h_stat.kurtosis))
def chest_measurements(patient, masks):

    middle_slice = masks[len(masks)//2]

    pixel_spacing = patient[0].PixelSpacing

    

    lung_area = np.round(np.sum(middle_slice.flatten()) * pixel_spacing[0] * pixel_spacing[1], 3)

    

    conv_h = morphology.convex_hull_image(middle_slice)

    props = measure.regionprops(measure.label(conv_h))

    

    chest_diameter = np.round(props[0].major_axis_length, 3)

    chest_circ = np.round(props[0].perimeter, 3)

    

    return lung_area, chest_diameter, chest_circ
ch_measure = chest_measurements(patient, masks)

print('Lung Area is: ' + str(ch_measure[0]) + ' mm^2')

print('Chest Diameter estimate is: ' + str(ch_measure[1]) + ' mm')

print('Chest Circumference estimate is: ' + str(ch_measure[2]) + ' mm')
#augmented_df = train_df.drop(train_df.loc[train_df['Patient'].isin(['ID00011637202177653955184', #

#                                                                    'ID00052637202186188008618'])]

#                             .index).reset_index(drop=True)
#augmented_df['LungVolume'] = None

#augmented_df['Mean'] = None

#augmented_df['Variance'] = None

#augmented_df['Skewness'] = None

#augmented_df['Kurtosis'] = None

#augmented_df['LungArea'] = None

#augmented_df['ChestDiameter'] = None

#augmented_df['ChestCircumference'] = None



#for pid in augmented_df['Patient'].unique():

#    patient_dir = '../input/osic-pulmonary-fibrosis-progression/train/' + pid

#    patient = load_scan(patient_dir)

#    scans = get_pixels_hu(patient)

    

#    segmented, masks = lung_segment_stack(scans)

    

#    augmented_df.loc[augmented_df['Patient'] == pid, 'LungVolume'] = lung_volume(patient, masks)

    

#    hist_stats = hist_analysis(segmented)

#    augmented_df.loc[augmented_df['Patient'] == pid, 'Mean'] = np.round(hist_stats.mean, 3)

#    augmented_df.loc[augmented_df['Patient'] == pid, 'Variance'] = np.round(hist_stats.variance, 3)

#    augmented_df.loc[augmented_df['Patient'] == pid, 'Skewness'] = np.round(hist_stats.skewness, 3)

#    augmented_df.loc[augmented_df['Patient'] == pid, 'Kurtosis'] = np.round(hist_stats.kurtosis, 3)

    

#    chest_stat = chest_measurements(patient, masks)

#    augmented_df.loc[augmented_df['Patient'] == pid, 'LungArea'] = chest_stat[0]

#    augmented_df.loc[augmented_df['Patient'] == pid, 'ChestDiameter'] = chest_stat[1]

#    augmented_df.loc[augmented_df['Patient'] == pid, 'ChestCircumference'] = chest_stat[2]
#plt.figure(figsize=(13,13))

#sns.heatmap(augmented_df.apply(lambda x : pd.factorize(x)[0] if x.dtype=='object' else x).corr(method='pearson'),

#            annot=True)
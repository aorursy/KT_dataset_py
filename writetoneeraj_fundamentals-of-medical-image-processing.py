%env JOBLIB_TEMP_FOLDER=/tmp
!conda install -c conda-forge gdcm -y
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.ndimage

from skimage import measure, morphology

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.cluster import KMeans

import shutil

import cv2

import pydicom

import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,

                               AutoMinorLocator)

import seaborn as sns

from IPython.display import HTML

import gdcm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def set_options():

    pd.set_option('display.max_columns', 100)

    pd.set_option('display.max_colwidth', None)

    pd.set_option('display.max_rows', 1000)
set_options()
TRAIN_PATH = '../input/osic-pulmonary-fibrosis-progression/train.csv'

TRAIN_IMG_PATH = '../input/osic-pulmonary-fibrosis-progression/train'

TEST_PATH = '../input/osic-pulmonary-fibrosis-progression/test.csv'

TEST_IMG_PATH = '../input/osic-pulmonary-fibrosis-progression/test'

SUBMISSION_PATH = '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv'
'''Load data'''

train = pd.read_csv(TRAIN_PATH)

test = pd.read_csv(TEST_PATH)
# Duplicates on basis of patient and weeks

train[train.duplicated(['Patient','Weeks'], keep=False)]
# Remove duplicates

train = train.drop_duplicates()
train.head()
HTML('<iframe width="600" height="400" src="https://www.youtube.com/embed/KZld-5W99cI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
## In this dataset ImagePosition is not available so we will sort the slices on InstanceNumber.

def load_slices(path):

    filenames = os.listdir(path)

    slices = [pydicom.dcmread(f'{path}/{file}') for file in filenames]

    slices.sort(key = lambda x: int(x.InstanceNumber), reverse=True)

    return slices
scans = load_slices(f'{TRAIN_IMG_PATH}/ID00007637202177411956430')

scans[0]
# Rescale intercept, (0028|1052), and rescale slope (0028|1053) are DICOM tags that specify the linear 

# transformation from pixels in their stored on disk representation to their in memory representation.

# Whenever the values stored in each voxel have to be scaled to different units, 

# Dicom makes use of a scale factor using two fields into the header 

# defining the slope and the intercept of the linear transformation to be used to 

# convert pixel values to real world values.



def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image <= -1000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)
def apply_window(hu_image, center, width):

    hu_image = hu_image.copy()

    min_value = center - width // 2

    max_value = center + width // 2

    hu_image[hu_image < min_value] = min_value

    hu_image[hu_image > max_value] = max_value

    return hu_image
train.loc[0]['Patient']
fig,ax = plt.subplots(1,2,figsize=(20,5))

example = train.loc[0]['Patient']

scans = load_slices(f'{TRAIN_IMG_PATH}/{example}')

rescaled_images=get_pixels_hu(scans)

images = [scan.pixel_array for scan in scans]

for i in range(10):

    sns.distplot(images[i].flatten(), ax=ax[0])

    sns.distplot(rescaled_images[i].flatten(), ax=ax[1])

ax[0].set_title("Raw pixel array distributions for 10 examples")

ax[1].set_title("HU unit distributions for 10 examples")
def get_dicom_raw(dicom):

    return ({attr:getattr(dicom, attr) for attr in dir(dicom) if attr[0].isupper() and attr not in ['PixelData']})
%%time

# Get dicom metadata

# Image features like lung volume are implementation from a detailed discussion "Domain expert's insight" by Dr. Konya.

# https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/discussion/165727



def get_dicom_metadata(df):

    patients = df.Patient.unique()

    dicom_metadata = []

    for patient in patients:

        path = f'{TRAIN_IMG_PATH}/{patient}'

        img_list = os.listdir(path)

        for img in img_list:

            image = pydicom.dcmread(f'{path}/{img}')

            record = get_dicom_raw(image)

            raw = image.pixel_array

            pixelspacing_r, pixelspacing_c = image.PixelSpacing[0], image.PixelSpacing[1]

            row_distance = pixelspacing_r * image.Rows

            col_distance = pixelspacing_c * image.Columns

            record.update({'raw_min':raw.min(),

                        'raw_max':raw.max(),

                        'raw_mean':raw.mean(),

                        'raw_std':raw.std(),

                        'raw_diff':raw.max()-raw.min(),

                        'pixel_spacing_area':pixelspacing_r * pixelspacing_c,

                        'img_area':image.Rows * image.Columns,

                        'pixel_row_distance':row_distance,

                        'pixel_col_distance':col_distance,

                        'slice_area_cm2':(0.1 * row_distance) * (0.1 * col_distance),

                        'slice_vol_cm3':(0.1 * image.SliceThickness) * (0.1 * row_distance) * (0.1 * col_distance),

                        'patient_img_path':f'{path}/{img}'})



            dicom_metadata.append(record)

            

    metadata_df = pd.DataFrame(dicom_metadata)

    metadata_df.to_pickle('metadata_df.pkl')

    return metadata_df
%%time

metadata_df = get_dicom_metadata(train.copy())

metadata_df.head()
plt.tight_layout()

fig, ax = plt.subplots(2, 2, figsize=(20,10))

sns.distplot(metadata_df.pixel_row_distance, ax=ax[0,0], color='green')

sns.distplot(metadata_df.pixel_col_distance, ax=ax[0,1], color='blue')

sns.distplot(metadata_df.slice_area_cm2, ax=ax[1,0], color='pink')

sns.distplot(metadata_df.slice_vol_cm3, ax=ax[1,1], color='magenta')

ax[0,0].set_title("Pixel Rows Distance")

ax[0,0].set_xlabel("Pixel Rows")

ax[0,1].set_title("Pixel Column Distance")

ax[0,1].set_xlabel("Pixel Columns")

ax[1,0].set_title("CT-slice area in $cm^{2}$")

ax[1,0].set_xlabel("Area in $cm^{2}$")

ax[1,1].set_title("CT-slice volume in $cm^{3}$")

ax[1,1].set_xlabel("Volume in $cm^{3}$")
# It is clearly visible that area and volume of lungs vary a lot. Let's show images with maximum volume and minimum volume.

highest_vol_patients = list(metadata_df[metadata_df.slice_vol_cm3 == max(metadata_df.slice_vol_cm3)]['PatientID'])

lowest_vol_patients = list(metadata_df[metadata_df.slice_vol_cm3 == min(metadata_df.slice_vol_cm3)]['PatientID'])

# Load scans for highest and lowest volume lung patients

max_vol_scans = load_slices(f"{TRAIN_IMG_PATH}/{highest_vol_patients[0]}")

min_vol_scans = load_slices(f"{TRAIN_IMG_PATH}/{lowest_vol_patients[0]}")

# Convert to HU

max_vol_hu_imgs = get_pixels_hu(max_vol_scans)

min_vol_hu_imgs = get_pixels_hu(min_vol_scans)

# Apply windowing]

# We can try with different window width and levels.

max_vol_window_img = apply_window(max_vol_hu_imgs[20], -600, 1200)

min_vol_window_img = apply_window(min_vol_hu_imgs[18], -600, 1200)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].imshow(max_vol_window_img, cmap="YlGnBu")

ax[0].set_title("CT with large volume")

ax[1].imshow(min_vol_window_img, cmap="YlGnBu")

ax[1].set_title("CT with small volume")
metadata_df.SliceThickness.unique()
# Lets see thickness for slices before thinking about resampling.

patient1 = train.Patient.unique()[0]

patient2 = train.Patient.unique()[5]

scans1 = load_slices(f"{TRAIN_IMG_PATH}/{patient1}")

scans2 = load_slices(f"{TRAIN_IMG_PATH}/{patient2}")

print(f"{scans1[0].SliceThickness}, {scans1[0].PixelSpacing}")

print(f"{scans2[0].SliceThickness}, {scans2[0].PixelSpacing}")

patient1_hu_scans = get_pixels_hu(scans1)

patient2_hu_scans = get_pixels_hu(scans2)
def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing

    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

    resize_factor = spacing / new_spacing

    new_real_shape = image.shape * resize_factor

    new_shape = np.round(new_real_shape)

    real_resize_factor = new_shape / image.shape

    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing
image1, rounded_new_spacing1 = resample(patient1_hu_scans, scans1, [1,1,1])

image2, rounded_new_spacing2 = resample(patient2_hu_scans, scans2, [1,1,1])

print(f"Original shape : {patient2_hu_scans.shape}")

print(f"Shape after resampling : {image2.shape}")
def plot_3d(image,threshold=800):

    

    # Position the scan upright, 

    # so the head of the patient would be at the top facing the   

    # camera

    p = image.transpose(2,1,0)

    

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of    

    # triangles

    mesh = Poly3DCollection(verts[faces], alpha=0.70)

    face_color = [0.45, 0.45, 0.75]

    mesh.set_facecolor(face_color)

    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])

    ax.set_ylim(0, p.shape[1])

    ax.set_zlim(0, p.shape[2])

    plt.show()
plot_3d(image1)
plot_3d(patient1_hu_scans)
#Standardize the pixel values

def make_lungmask(img, display=False):

    row_size= img.shape[0]

    col_size = img.shape[1]

    mean = np.mean(img)

    std = np.std(img)

    img = img-mean

    img = img/std

    

    # Find the average pixel value near the lungs

    # to renormalize washed out images

    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 

    mean = np.mean(middle)  

    max = np.max(img)

    min = np.min(img)

    # To improve threshold finding, I'm moving the 

    # underflow and overflow on the pixel spectrum

    img[img==max]=mean

    img[img==min]=mean

    

    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)

    #

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))

    centers = sorted(kmeans.cluster_centers_.flatten())

    threshold = np.mean(centers)

    

    # Threshold the image and the output will be a binary image. Morphology workes either on binary or gray images.

    thresh_img = np.where(img<threshold,1.0,0.0)

    

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  

    # We don't want to accidentally clip the lung.



    eroded = morphology.erosion(thresh_img,np.ones([3,3]))

    dilation = morphology.dilation(eroded,np.ones([8,8]))



    labels = measure.label(dilation) # Different labels are displayed in different colors

    label_vals = np.unique(labels)

    regions = measure.regionprops(labels)

    good_labels = []

    for prop in regions:

        B = prop.bbox

        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:

            good_labels.append(prop.label)

    mask = np.ndarray([row_size,col_size],dtype=np.int8)

    mask[:] = 0



    #  After just the lungs are left, we do another large dilation

    #  in order to fill in and out the lung mask 

    for N in good_labels:

        mask = mask + np.where(labels==N,1,0)

    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation



    if (display):

        fig, ax = plt.subplots(3, 2, figsize=[12, 12])

        ax[0, 0].set_title("Original")

        ax[0, 0].imshow(img, cmap='gray')

        ax[0, 0].axis('off')

        ax[0, 1].set_title("Threshold")

        ax[0, 1].imshow(thresh_img, cmap='gray')

        ax[0, 1].axis('off')

        ax[1, 0].set_title("After Erosion and Dilation")

        ax[1, 0].imshow(dilation, cmap='gray')

        ax[1, 0].axis('off')

        ax[1, 1].set_title("Color Labels")

        ax[1, 1].imshow(labels)

        ax[1, 1].axis('off')

        ax[2, 0].set_title("Final Mask")

        ax[2, 0].imshow(mask, cmap='gray')

        ax[2, 0].axis('off')

        ax[2, 1].set_title("Apply Mask on Original")

        ax[2, 1].imshow(mask*img, cmap='gray')

        ax[2, 1].axis('off')

        

        plt.show()

    return mask*img
make_lungmask(image1[14], True)
def get_rows_cols(size):

    cols = 6

    rows = size // cols

    if (int(size%cols) != 0):

        rows = rows+1

    return rows,cols
def plot_stack(stack, start_with=10, show_every=3):

    size = (len(stack) - (start_with - 1))//show_every

    rows, cols = get_rows_cols(size)

    plt.tight_layout()

    fig,ax = plt.subplots(rows,cols,figsize=[12,12])

    for i in range(size-1):

        ind = start_with + i*show_every

        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)

        ax[int(i/cols),int(i % cols)].imshow(stack[ind],cmap='gray')

        ax[int(i/cols),int(i % cols)].axis('off')

    plt.show()
plot_stack(patient1_hu_scans, start_with=0, show_every=1)
masked_lung = []



for img in image1:

    masked_lung.append(make_lungmask(img))

    

plot_stack(masked_lung, start_with=0, show_every=1)
# Code for running processing on the whole data all together. Outcome of the code will 



# be .npz file for all patients. .npz files can be loaded using np.load() function for further use.

# If you want to store images in .png files remove the comments from below code and comment out code mentioned below.

'''

path = "./segmented-images"

if not shutil.os.path.isdir(path):

    shutil.os.mkdir(path)



patients = train.Patient.unique()[0:10]

for patient in patients:

    #if not shutil.os.path.isdir(path + "/" + patient):

    #    shutil.os.mkdir(path + "/" + patient)

    scans = load_slices(f'{TRAIN_IMG_PATH}/{patient}')

    hu_imgs = get_pixels_hu(scans)

    rescaled_images, spacing = resample(hu_imgs, scans,[1,1,1])



    masked_lung = []

    for img_number in range(len(rescaled_images)):

        window_img = apply_window(rescaled_images[img_number], -600, 1200)

        masked_img = make_lungmask(window_img)

        masked_lung.append(masked_img)

        #cv2.imwrite(f'{path}/{patient}/{img_number + 1}.png', masked_img)

    # Comment the below line if images required to store in .png format.

    np.savez(f'{path}/{patient}',masked_lung)

    #plot_stack(masked_lung, start_with=0, show_every=1)

'''
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

from os import listdir





import plotly.express as px

import plotly.graph_objs as go





import pydicom

import glob

import imageio

from IPython.display import Image





import warnings

warnings.filterwarnings('ignore')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    i=0

    for filename in filenames:

        i+=1        

        print(os.path.join(dirname, filename))

print(str(i))

len(os.listdir('/kaggle/input/osic-pulmonary-fibrosis-progression/train/'))
list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
traindf = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")

testdf = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

sample_submissiondf = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
traindf.describe()
traindf.info()
testdf.info()
testdf.describe()
traindf.head()
traindf['Patient'].count()
patients_with_multiple_data = traindf["Patient"].value_counts()



patients_with_multiple_data.count()

patients_with_multiple_data.describe()
new_df = traindf.groupby([traindf.Patient,traindf.Age,traindf.Sex, traindf.SmokingStatus])['Patient'].count()

new_df.index = new_df.index.set_names(['id','Age','Sex','SmokingStatus'])

new_df = new_df.reset_index()

new_df.rename(columns = {'Patient': 'freq'},inplace = True)

new_df.head(5)
new_df.shape
fig = px.bar(new_df, x="id", y="freq", color='freq')

fig.update_layout(xaxis={'categoryorder':'total ascending'},title='number of data entries for each patient')

fig.update_xaxes(showticklabels=False)

fig.show()
hst = px.histogram(new_df, x='Age', nbins=40, opacity=0.7,title='Age of patients against the number of records', labels={'Age':'Age of patients', 'freq':'Records available for each patient'})

hst.update_traces(marker_color='rgb(123,125,222)', marker_line_color='rgb(9,4,21)', marker_line_width=1.5)

hst.show()
agehst = px.histogram(new_df, x='Sex')

agehst.show()
sexdf = new_df['Sex'].value_counts()

total = new_df.shape[0]

male = sexdf['Male']



female = sexdf['Female']

print('Out of %d patients' % total)

print('%.2f percent are male' % (male/total))

print('%.2f percent are females' % (female/total))
image = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00060637202187965290703/107.dcm'

ds = pydicom.dcmread(image)

plt.figure(figsize=(12,12))

plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
imgdir = '/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00060637202187965290703/'

img_list = os.listdir(imgdir)

img_list.sort()

len(img_list)





fig=plt.figure(figsize=(15,20))

columns = 5

rows = 10



for i in range(1, columns*rows +1):

    filename = imgdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
def load_scan(path):

    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices





def get_pixels_hu(slices):

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16), 

    # should be possible as values should always be low enough (<32k)

    image = image.astype(np.int16)



    # Set outside-of-scan pixels to 0

    # The intercept is usually -1024, so air is approximately 0

    image[image == -2000] = 0

    

    # Convert to Hounsfield units (HU)

    for slice_number in range(len(slices)):

        

        intercept = slices[slice_number].RescaleIntercept

        slope = slices[slice_number].RescaleSlope

        

        if slope != 1:

            image[slice_number] = slope * image[slice_number].astype(np.float64)

            image[slice_number] = image[slice_number].astype(np.int16)

            

        image[slice_number] += np.int16(intercept)

    

    return np.array(image, dtype=np.int16)



def set_lungwin(img, hu=[-1200., 600.]):

    lungwin = np.array(hu)

    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])

    newimg[newimg < 0] = 0

    newimg[newimg > 1] = 1

    newimg = (newimg * 255).astype('uint8')

    return newimg

scans = load_scan('../input/osic-pulmonary-fibrosis-progression/train/ID00060637202187965290703/')

scan_array = set_lungwin(get_pixels_hu(scans))







imageio.mimsave("/tmp/gif.gif", scan_array, duration=0.0001)

Image(filename="/tmp/gif.gif", format='png')
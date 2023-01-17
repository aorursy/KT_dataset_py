import sys

import os

import subprocess



from six import string_types



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import scipy

from skimage import io

from scipy import ndimage

from IPython.display import display

%matplotlib inline

! ls -lrt ../input
PLANET_KAGGLE_ROOT = os.path.abspath('../input/')

PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)

assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)

assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)            
!ls -lha /kaggle/input/train_v2.csv
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

labels_df.head()
label_list = []

for tag_str in labels_df.tags.values:

    labels = tag_str.split(' ')

    for label in labels:

        if label not in label_list:

            label_list.append(label)

label_list

#labels_df['tags']
for label in label_list:

    labels_df[label] =  labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0 )

  

labels_df.head()
labels_df[label_list].sum().sort_values().plot.bar()
def make_cooccurrence_matrix(labels):

    numeric_df = labels_df[labels]

    c_matrix = numeric_df.T.dot(numeric_df)

    sns.heatmap(c_matrix)

    return c_matrix



make_cooccurrence_matrix(label_list)
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']

make_cooccurrence_matrix(weather_labels)
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']

make_cooccurrence_matrix(land_labels)
rare_labels = [l for l in label_list if labels_df[label_list].sum()[l] < 2000]

make_cooccurrence_matrix(rare_labels)
def sample_images(tags, n=None):

    ''' Randomly sample n images with the specified tags.'''

    condition = True

    if isinstance(tags, string_types):

        raise ValueError('Pass a list of tags, not a single tag')

    for tag in tags:

        condition = condition & labels_df[tag] == 1

    if n is not None:

        return labels_df[condition].sample(n)

    else:

        return labels_df[condition]

    
def load_image(filename):

    '''Look through directory tree to find the image specified'''

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            print('Found image {}'.format(path))

            return io.imread(path)

        print('Load failed: could not find image {}'.format(path))

        

def sample_to_fname(sample_df, row_idx, suffix='tif'):

    '''Given a dataframe of sampled images, get the corresponding filename'''

    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')

    return '{}.{}'.format(fname, suffix)
def plot_rgbn_hist(r, g, b, n):

    for slice_, name, color in ((r, 'r', 'red'), (g, 'g', 'green'), (b, 'b', 'blue'), (n, 'n', 'magenta')):

        plt.hist(slice_.ravel(), bins=100,

                range=[0,rgb_image.max()],

                label=name, color=color,histtype='step')

    plt.legend()    
s = sample_images(['primary', 'water', 'road'], n=1)

fname = sample_to_fname(s, 0)



bgrn_image = load_image(fname)



bgr_image = bgrn_image[:,:,:3]

rgb_image = bgr_image[:, :, [2,1,0]]



# Extract each band

# extract the different bands

b, g, r, nir = bgrn_image[:, :, 0], bgrn_image[:, :, 1], bgrn_image[:, :, 2], bgrn_image[:, :, 3]



# plot a histogram of rgbn values

plot_rgbn_hist(r, g, b, nir)
# Plot the bands

fig = plt.figure()

fig.set_size_inches(12, 4)

for i, (x, c) in enumerate(((r, 'r'), (g, 'g'), (b, 'b'), (nir, 'near-ir'))):

    a = fig.add_subplot(1, 4, i+1)

    a.set_title(c)

    plt.imshow(x)                       
plt.imshow(rgb_image)
# Collect a list of 20000 image names

jpg_list = os.listdir(PLANET_KAGGLE_JPEG_DIR)[:20000]



# Pick 100 at random

np.random.shuffle(jpg_list)

jpg_list = jpg_list[:100]



print(jpg_list)
ref_colors = [[],[],[]]

for _file in jpg_list:

    #keep only rgb

    _img = mpimg.imread(os.path.join(PLANET_KAGGLE_JPEG_DIR, _file))[:,:,:3]

    #flatten the 2 dimensions to one

    _data = _img.reshape((-1,3))

    # Dump pixel values into correct buckets

    for i in range(3):

        ref_colors[i] = ref_colors[i] + _data[:,i].tolist()

    

ref_colors = np.array(ref_colors)    
for i, color in enumerate(['r','g','b']):

    plt.hist(ref_colors[i], bins=30, range=[0,255], label=color, color=color, histtype='step')

plt.legend()

plt.title('Reference colour histogram')

   
ref_means = [np.mean(ref_colors[i]) for i in range(3)]

ref_stds = [np.std(ref_colors[i]) for i in range(3)]
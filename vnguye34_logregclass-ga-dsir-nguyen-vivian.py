# some notes

# I found this dataset from https://www.reddit.com/r/datasets/comments/jd788k/corn_leaf_infection_dataset_taken_from_field/
# As of starting this notebook, this dataset has only been out for 2 days. (19/10/2020)
# This serves as a Logistic Regression Benchmark for future classification projects for this dataset

# If you can use my pre processing pipeline or logic, please cite your source back to this notebook
# Let's help farmers and their corn crops!
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import matplotlib
import matplotlib.pyplot as plt

# Load images as arrays
import PIL

# Filenames via Object Generators
from pathlib import Path

# Python Generators
import itertools
from itertools import chain

#Sklearn image preprocessing
from skimage.transform import rescale

# Machine Learning
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Metrics
from sklearn.metrics import plot_confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
'''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
''';
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_raw = pd.read_csv('/kaggle/input/corn-leaf-infection-dataset/Annotation-export.csv')
df = data_raw.copy()
df.head()
df.info()
df['label'].value_counts()
'/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Infected/'+df.head().loc[0, 'image']
filenamelist_healthy_generator = Path('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Healthy corn/').glob('*.jpg')
healthy_corn_filenames = [i for i in filenamelist_healthy_generator]
plt.figure(figsize=(20,15))
gs1 = matplotlib.gridspec.GridSpec(2, 4)
gs1.update(wspace=0.2, hspace=0.2)
unique_infected_images = df['image'].unique()
for i in range(1,5):
    plt.subplot(2,4,i)  
    img = matplotlib.image.imread('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Infected/'+unique_infected_images[i])
    plt.imshow(img)
    plt.title('Infected')
    currentAxes = plt.gca()
    
    df_sorted = df[df['image'] == unique_infected_images[i]]
    
    for j in [df_sorted.reset_index(drop=True).loc[i, ['xmin', 'ymin', 'xmax', 'ymax']] for i in range(len(df_sorted))]:
        currentAxes.add_patch(matplotlib.patches.Rectangle(xy=(j['xmin'], j['ymin']), 
                                                           height=abs(j['ymax'] - j['ymin']),
                                                           width=abs(j['xmax'] - j['xmin']), 
                                                           color='red', 
                                                           linewidth=2, 
                                                           fill=False))
        plt.scatter(x=[j['xmin'], j['xmax']], y=[j['ymin'], j['ymax']], c='r')
    #plt.show()

for i in range(1,5):
    plt.subplot(2, 4,i+4)
    img = matplotlib.image.imread(healthy_corn_filenames[i])
    plt.imshow(img)
    plt.title('Healthy');
    
plt.tight_layout()
plt.savefig('/kaggle/working/example_corn.png', bbox_inches='tight', pad_inches=0)
# From here on, we are going to be focusing on the easier of the two tasks of the dataset

# Find a best model to classify Infected vs Healthy leaves

# We are going to use a variety of statistical models to do a bernoulli classification problem: 
    # https://scikit-learn.org/0.15/_images/plot_classifier_comparison_0011.png
    
    # 0: There is not disease on the leaf
    # 1: There is disease on the leaf
    
# What am I going to use?
    # Sklearn Classification: https://scikit-learn.org/0.15/_images/plot_classifier_comparison_0011.png
    # We can consider the BGR image as a (width x height x color) tensor array. 
    # I believe that color is a large indicator of detecting disease so I'm not going to go down to grayscale
    
    # To input the image, we need to reduce them down. 
    # Thankfully, because these were all taken with the same photo, they are all the same aspect ratio and I don't need to worry about skewing or shearing them
        # See here: https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
        
        # Normalize:
            # I am going to need to rescale these tensors from (0, 255) to (0, 1) to handle preprocessing
                # Source: experience. I already know I am going to get improved preformance by scaling
                
        # Centering:
            # I want to subtract the mean value to center the data around 0
            # Source: Experience. I've found that by normalizing and centering, we lose no information of what was in the image but
                # Instead give our estimator a much easier time in processing through the data by not dealing with extremely large and small numbers
                
                # Note, there may be some merit in comparing centering before and after normalization. We'll take that into account here
                
                # I am going to do local centering per image, because I think there is value in each matrix being an instance and I am trying to find differences
                # in each matrix.
# Let's test it on one image. We're going to use the first infected image we saw above
# Load it in with PIL, convert to an dense numpy array, see the first pixel in format RGB
test_filename = '/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Infected/'+unique_infected_images[0]
test_image = PIL.Image.open(test_filename)
image_data = np.asarray(test_image)
image_data[0][0]
sample_data_raw = image_data[0][0:1]
sample_data = sample_data_raw
# This is a visualization of the standardization methods
plt.figure(figsize=(25,20))
# Nothing
plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data_raw[0]):
    plt.subplot(6,3,i+1)
    plt.imshow(sample_data_raw, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')

#Normalize
sample_data = Normalizer().fit_transform(sample_data_raw)
#sample_data = StandardScaler().fit_transform(sample_data)

plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data[0]):
    plt.subplot(6,3,i+4)
    plt.imshow(sample_data, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')

# Scale
sample_data = StandardScaler().fit_transform(sample_data_raw.T)

plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data.T[0]):
    plt.subplot(6,3,i+7)
    plt.imshow(sample_data.T, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')

#Normalize & Scale, Order matters

sample_data = Normalizer().fit_transform(sample_data_raw)
sample_data = StandardScaler().fit_transform(sample_data.T)
cmap=plt.cm.gray
plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data.T[0]):
    plt.subplot(6,3,i+10)
    plt.imshow(sample_data.T, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')

#Scale and Normalize, Order matters

sample_data = StandardScaler().fit_transform(sample_data_raw.T)
sample_data = Normalizer().fit_transform(sample_data)

plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data.T[0]):
    plt.subplot(6,3,i+13)
    plt.imshow(sample_data.T, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')

#Scale and Normalize, Order matters

sample_data = StandardScaler().fit_transform(sample_data_raw.T)
sample_data = Normalizer().fit_transform(sample_data)

plt.figure(figsize=(15,8))
color=['Reds', 'Blues', 'Greens']
for i, j in enumerate(sample_data.T[0]):
    plt.subplot(6,3,i+16)
    plt.imshow(sample_data.T, cmap=color[i])
    plt.title(j, color='black', fontsize=15, ha='center')
# Okay, so when we are going to do testing on this data, we seriously want smaller pictures
plt.imshow(image_data)
plt.imshow((rescale(image_data[:, :, 1], 0.1, anti_aliasing=True)))
def corn_preprocessing(file_list, n):
    '''
    Takes in a list of file names
    
    Applys preprocessing to the image files in two sets:
    
    Set 0: Nothing, the orignal dataframe is passed through for comparison
    Set 1: (Normalize, Centering)
    Set 2: (Centering, Normalize)
    ''';
    # Normalizer moves the range of our data from 0 - 1
    # Standard Scaler removes the mean and scales to unit variance
    print(file_list[0])
    
    # sourced from PIL docs
    image_data = [np.mean(np.asarray(i.resize((300,300))), axis=2) for i in (PIL.Image.open(i) for i in file_list[:n])] # This is hard coded for preformance, # changed to output grayscale
    
    gen_matrix = image_data
    gen_matix_nn_ss = (StandardScaler().fit_transform(X=Normalizer().fit_transform(i)) for i in image_data)
    gen_matrix_ss_nn = (Normalizer().fit_transform(X=StandardScaler().fit_transform(i)) for i in image_data)
    
    return gen_matrix, gen_matix_nn_ss, gen_matrix_ss_nn
# Get our transformed matricies of the infected images
inf_normal, inf_norm_ss, inf_ss_norm = corn_preprocessing('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Infected/'+df['image'],100)
# Let's run a test
# Get our list of filenames of healthy leaves
filenamelist_healthy = [i for i in Path('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Healthy corn/').glob('*.jpg')]
# Transform the list of healthy filenmes 
hea_normal, hea_norm_ss, hea_ss_norm = corn_preprocessing([str(i) for i in filenamelist_healthy],100)
# Okay, now we have a bunch of generators to lazy load our data. as we can't load 13GB * 3 into memory.
# What we can do is consider a first instance where we train our classifier on some part of the data, let's say a train test split of 20

# Here is a cool trick to combine generators
# https://stackoverflow.com/questions/3211041/how-to-join-two-generators-in-python

# We need to add labels to our data. We know that everything for inf is infected. We can set a boolean map of [0,1] which represents the labels for everything in that set
#y = np.append(np.zeros(len(df)),np.ones(len(filenamelist_healthy)))

#This has been hardcoded for only 100 entries
y = np.append(np.zeros(100),np.ones(100))
y.shape
# Let's set up a simple classifier class so we can handle multiple estimators and have a consistent train test split across the data when changing estimators
class corn_classifier():
    '''
    This classifier class is based off of my previous work, found here:
    https://github.com/vnguye34/dsfunctions/blob/master/dsfunctions/pipelines.py
    '''
    
    def __init__(self, X, y):
        '''
        Sets up universal train test split for inputted data
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        
        # Sourced from Numpy docs for combining an array of arrays
        self.X_train = np.concatenate([i.reshape(1,-1) for i in X_train] , axis=0)
        self.X_test = np.concatenate([i.reshape(1,-1) for i in X_test], axis=0)
        self.y_train = y_train
        self.y_test = y_test
        
    def fit_estimator(self, estimator_object):
        '''
        Fits X train, y_train to the object
        Returns object and the accuracy score for train and test datasets
        '''
        fitted_estimator = estimator_object.fit(self.X_train, self.y_train)
        train_accuracy_score = estimator_object.score(self.X_train, self.y_train)
        test_accuracy_score = estimator_object.score(self.X_test, self.y_test)
        
        return fitted_estimator, train_accuracy_score, test_accuracy_score
# Instantiate the class object
cc = corn_classifier(X =  [i for i in chain(inf_normal, hea_normal)], y = y)
plt.imshow(cc.X_train[0].reshape(300,300), cmap=plt.cm.gray)
cc.X_train.shape
cc.y_train.shape
logr = LogisticRegression()
fitted_logr, train_acc, test_acc = cc.fit_estimator(estimator_object=logr)
pd.Series(y).value_counts(normalize=True)
train_acc, test_acc
# This seems already too good to be true
plot_confusion_matrix(fitted_logr, X = cc.X_test, y_true=cc.y_test)
# I really don't trust how well this model did on 100 samples. See below for a more generalizable model I've attempted
# NOTE: some additional optimization work needs to be done in order to keep progressing

# Build a bacthing dataloader OR use an exising one from Tensorflow or PyTorch
# Get our transformed matricies of the infected images
inf_normal_2, inf_norm_ss_2, inf_ss_norm_2 = corn_preprocessing('/kaggle/input/corn-leaf-infection-dataset/Corn Disease detection/Infected/'+df['image'], 1000)

# Transform the list of healthy filenmes 
hea_normal_2, hea_norm_ss_2, hea_ss_norm_2 = corn_preprocessing([str(i) for i in filenamelist_healthy], 1000)

# set up y
y_2 = np.append(np.zeros(1000),np.ones(1000))

# Instantiate the class object
cc_2 = corn_classifier(X =  [i for i in chain(inf_normal_2, hea_normal_2)], y = y_2)
cc_2_nn_ss = corn_classifier(X =  [i for i in chain(inf_norm_ss_2, hea_norm_ss_2)], y = y_2)
cc_2_ss_nn = corn_classifier(X =  [i for i in chain(inf_ss_norm_2, hea_ss_norm_2)], y = y_2)


logr_2 = LogisticRegression(max_iter=100_000)
fitted_logr_2, train_acc_2, test_acc_2 = cc_2.fit_estimator(estimator_object=logr_2)
fitted_logr_2_nn, train_acc_2_nn, test_acc_2_nn = cc_2.fit_estimator(estimator_object=logr_2)
fitted_logr_2_ss, train_acc_2_ss, test_acc_2_ss = cc_2.fit_estimator(estimator_object=logr_2)
plot_confusion_matrix(fitted_logr_2, X = cc.X_test, y_true=cc.y_test)
plot_confusion_matrix(fitted_logr_2_nn, X = cc.X_test, y_true=cc.y_test)
plot_confusion_matrix(fitted_logr_2_ss, X = cc.X_test, y_true=cc.y_test)

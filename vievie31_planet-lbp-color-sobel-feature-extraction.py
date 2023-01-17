import os

import sys



import scipy

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



from skimage import io

from scipy.ndimage import convolve

from skimage.transform import rotate

from skimage.feature import local_binary_pattern



from scipy import ndimage

from scipy.stats import itemfreq 



from sklearn.utils import shuffle

from sklearn.metrics import precision_recall_fscore_support

from sklearn.feature_extraction.image import extract_patches_2d

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn import linear_model, decomposition

from sklearn.cluster import MiniBatchKMeans



from pprint import pprint

from IPython.display import display

from itertools import product, chain



%matplotlib inline



PLANET_KAGGLE_ROOT = os.path.abspath("../input/")



#training sets paths

TRAIN_JPEG_DIR  = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')

TRAIN_TIF_DIR   = os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2')

TRAIN_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')



assert os.path.exists(TRAIN_JPEG_DIR)

assert os.path.exists(TRAIN_TIF_DIR)

assert os.path.exists(TRAIN_LABEL_CSV)



#testing sets paths

TEST_JPEG_DIR  = os.path.join(PLANET_KAGGLE_ROOT, 'test-jpg-v2')

TEST_TIF_DIR   = os.path.join(PLANET_KAGGLE_ROOT, 'test-tif-v2')



assert os.path.exists(TEST_JPEG_DIR)

assert os.path.exists(TEST_TIF_DIR)
labels_df = pd.read_csv(TRAIN_LABEL_CSV)



#build list with unique labels

label_list = []

for tag_str in labels_df.tags.values:

    labels = tag_str.split(' ')

    for label in labels:

        if label not in label_list:

            label_list.append(label)



#qdd onehot features for every label

for label in label_list:

    labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)



#remove the tags column

labels_df = labels_df.drop("tags", 1)



labels_df.head()
def load_image(filename):

    '''Look through the directory tree to find the image you specified

    (e.g. train_10.tif vs. train_10.jpg)'''

    for dirname in os.listdir(PLANET_KAGGLE_ROOT):

        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))

        if os.path.exists(path):

            #print('Found image {}'.format(path))

            return io.imread(path)

    #print('Load failed: could not find image {}'.format(path))



def sample_to_fname(sample_df, row_idx, suffix='tif'):

    '''Given a dataframe of sampled images, get the

    corresponding filename.'''

    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')

    return '{}.{}'.format(fname, suffix)
def extract_rgb(img): 

    return img[:,:,0], img[:,:,1], img[:,:,2]



def rgb2gray(rgb):

    r, g, b = extract_rgb(rgb)

    return 0.2989 * r + 0.5870 * g + 0.1140 * b



LBP_RADIUS = 3

LBP_NB_POINTS = 8 * LBP_RADIUS

LBP_METHOD = 'uniform'



def get_lbp(img_gray_scale):

    return local_binary_pattern(

        img_gray_scale,

        LBP_NB_POINTS,

        LBP_RADIUS,

        LBP_METHOD

    )



def get_hist(lst):

    x = itemfreq(lst)

    hist = x[:, 1]/sum(x[:, 1])

    return hist



def get_lbp_hist(img):

    return get_hist(get_lbp(rgb2gray(img)).reshape(-1))



##load and display image

#img = load_image("train_1.jpg")

#r, g, b = extract_rgb(img)

#plt.imshow(get_lbp(rgb2gray(img)), cmap='gray')

#plt.show()



##compute the lbp histogram of the image

#hist = get_lbp_hist(img)

#plt.plot(range(len(hist)), hist)

#plt.show()



##compute the lbp histogram of the image rotated by 30°

#hist_r30 = get_lbp_hist(rotate(img, angle=30, resize=False))

#plt.plot(range(len(hist_r30)), hist_r30)

#plt.show()
def get_sobel(img_gray_scale):

    sx = ndimage.sobel(img_gray_scale, axis=0, mode='constant')

    sy = ndimage.sobel(img_gray_scale, axis=1, mode='constant')

    return np.hypot(sx, sy)[3:-3,3:-3] #3:-3 to remove the border problems...



def get_sobel_hist(img):

    x = np.histogram(get_sobel(rgb2gray(img)).reshape(-1), bins=range(256))[0]

    return x / sum(x)



##load image

#img = load_image("train_11.jpg")



##display the sobel fitered image

#plt.imshow(get_sobel(rgb2gray(img)))

#plt.show()



##disply the sobel histogram

#sobel_hist = get_sobel_hist(img)

#plt.plot(range(len(sobel_hist)), sobel_hist)

#plt.show()
def extract_gray_scale_histogram(img_gray_scale):

    x = np.histogram(img_gray_scale.reshape(-1), bins=range(256))[0]

    return x / sum(x)



def get_color_histogram(img):

    f = lambda m: list(extract_gray_scale_histogram(m))

    r, g, b = extract_rgb(img)

    return f(r) + f(g) + f(b)



#load image

#img = load_image("train_12.jpg")



#display image

#plt.imshow(img)

#plt.show()



#display color histogram

#plt.plot(get_color_histogram(img))

#plt.show()
PRC_TRAINING = .5

PRC_TESTING  = .25



dataset = shuffle(labels_df).reset_index(drop=True)



train, test = train_test_split(dataset, test_size=1 - PRC_TRAINING)

test, cv = train_test_split(dataset, test_size=PRC_TESTING)



training_set_labels = train

testing_set_labels  = test

cross_validation_set_labels = cv
LOAD_MAX_NB_DATA = 1000



#loading the data

training_set_data = []

testing_set_data  = []

cross_validation_set_data = []



for r in training_set_labels["image_name"][:LOAD_MAX_NB_DATA]:

    img = load_image(r + ".jpg")

    training_set_data.append(img)#extract_features(img))

print("training data loaded...")



for r in testing_set_labels["image_name"][:LOAD_MAX_NB_DATA]:

    img = load_image(r + ".jpg")

    testing_set_data.append(img)#extract_features(img))

print("testing data loaded...")

    

for r in cross_validation_set_labels["image_name"][:LOAD_MAX_NB_DATA]:

    img = load_image(r + ".jpg")

    cross_validation_set_data.append(img)#extract_features(img))

print("cross validation data loaded...")

    

#removing the image name column in labels

training_set_labels = training_set_labels.drop("image_name", 1)

testing_set_labels  = testing_set_labels.drop("image_name", 1)

cross_validation_set_labels = cross_validation_set_labels.drop("image_name", 1)



#limit the the nb max data to load...

training_set_labels = training_set_labels.head(LOAD_MAX_NB_DATA)

testing_set_labels  = testing_set_labels.head(LOAD_MAX_NB_DATA)

cross_validation_set_labels = cross_validation_set_labels.head(LOAD_MAX_NB_DATA)
DISPLAY_CLUSTER_MATRIX = (9, 9)

kmeans = MiniBatchKMeans(n_clusters=DISPLAY_CLUSTER_MATRIX[0] * DISPLAY_CLUSTER_MATRIX[1], verbose=False)

patch_size = (20, 20)



buffer = []

index = 1



for img in cross_validation_set_data: #using the crossvalidation dataset to train the words extractor...

    data = extract_patches_2d(rgb2gray(img), patch_size, max_patches=500)

    data = np.reshape(data, (len(data), -1))

    index += 1

    buffer.append(data)

    data = np.concatenate(buffer, axis=0)



print("nb_patches : ", len(buffer))

    

kmeans.fit(data)
#display the patches

plt.figure(figsize=(10.2, 10))

for i, patch in enumerate(kmeans.cluster_centers_):

    plt.subplot(DISPLAY_CLUSTER_MATRIX[0], DISPLAY_CLUSTER_MATRIX[1], i + 1)

    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray, interpolation='nearest')

    plt.xticks(())

    plt.yticks(())



plt.suptitle('Patches trained', fontsize=16)

plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)



plt.show()
def get_words_hist_of_img(gray_scale_img):

    #initialise the vector

    hist = [0 for i in range(kmeans.n_clusters)]

    #extract patches

    data = extract_patches_2d(gray_scale_img, patch_size, max_patches=500)

    data = np.reshape(data, (len(data), -1))

    #comoute the predictions histogram

    preds = kmeans.predict(data)

    for p in preds:

        hist[p] += 1

    hist = np.array(hist)

    return hist / sum(hist)

    

#get_words_hist_of_img(rgb2gray(img))
def extract_features(img):

    lst = list(get_lbp_hist(img)) + list(get_sobel_hist(img)) + list(get_color_histogram(img)) + list(get_words_hist_of_img(rgb2gray(img)))

    #print(len(list(get_lbp_hist(img))), len(list(get_sobel_hist(img))), len(list(get_color_histogram(img))))

    return lst



#load image

#img = load_image("train_12.jpg")



#display the global descripor

#plt.plot(extract_features(img))

#plt.show()
training_set_img = training_set_data

testing_set_img  = testing_set_data



training_set_data = []

testing_set_data  = []



for img in training_set_img:

    training_set_data.append(extract_features(img))

print("training features extracted...")



for img in testing_set_img:

    testing_set_data.append(extract_features(img))

print("testing features extracted...")
LR = lambda: linear_model.LogisticRegression(class_weight="balanced", penalty="l2")

PCA = lambda: decomposition.PCA()

PIPE_LINE = lambda: Pipeline(steps=[

    #('pca', PCA()), 

    ('logistic', LR())

])



##TODO: configure crossvalidated logistic regression...

#LRCV = lambda: linear_model.LogisticRegressionCV(class_weight="balanced", penalty="l2", cv=StratifiedKFold, scoring=precision_recall_fscore_support)



#fit logistic models

predictors = {label: PIPE_LINE() for label in list(training_set_labels)}

for label in predictors:

    try:

        predictors[label].fit(

            np.array(training_set_data),

            np.array(list(training_set_labels[label]))

        )

    except:

        print("something went wrong with label : ", label)





#compute model accuracy    

accuracy = {}

for label in predictors:

    try:

        y_hat = predictors[label].predict(testing_set_data) == np.array(list(testing_set_labels[label]))

        accuracy[label] = sum(y_hat) / len(y_hat)

    except:

        print("ignoring label : ", label)



print("accuracy : ")

pprint(accuracy)
#####TODO build the top level classifier...

## adapt the previous classifier set to ouput a set of labels 

## (think about uniques labels eg. wheater can't be cloudy and partialy cloudy...)
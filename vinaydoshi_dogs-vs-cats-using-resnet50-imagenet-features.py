# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import h5py

import pickle

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import classification_report, accuracy_score

from sklearn.linear_model import LogisticRegression

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# /kaggle/input/resnet50-imagenet-weights-dogs-vs-cats/resnet_dogs_vs_cats_features_v2.hdf5

# /kaggle/input/resnet50-imagenet-weights-dogs-vs-cats/resnet_dogs_vs_cats_testset_features.hdf5

# print(features.shape, labels.shape)
db = '../input/resnet50-imagenet-weights-dogs-vs-cats/resnet_dogs_vs_cats_features_v2.hdf5'

db = h5py.File(db, 'r')

features = db.get('features').value

labels =  db.get('labels').value#.reshape(25000,-1)

print(features.shape, labels.shape)
#db['features'][:].shape

label_names = db.get('label_names').value
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, stratify=labels)
params = [{'C':[0.0001, 0.001, 0.01, 0.1, 1, 10]

          }]

logreg = LogisticRegression(n_jobs=-1)

grid = GridSearchCV(estimator=logreg, param_grid=params, cv=3, n_jobs=-1, verbose = 2)

grid.fit(X_train, y_train)
print(grid.best_params_)

model_logreg = grid.best_estimator_

model_logreg
preds = model_logreg.predict(X_test)

print('Accuracy Score:',accuracy_score(y_test, preds))

print(classification_report(y_test, preds, target_names=label_names))
model_logreg.fit(features,labels)
#db.close()
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")





def list_images(basePath, contains=None):

    # return the set of files that are valid

    return list_files(basePath, validExts=image_types, contains=contains)





def list_files(basePath, validExts=None, contains=None):

    # loop over the directory structure

    for (rootDir, dirNames, filenames) in os.walk(basePath):

        # loop over the filenames in the current directory

        for filename in filenames:

            # if the contains string is not none and the filename does not contain

            # the supplied string, then ignore the file

            if contains is not None and filename.find(contains) == -1:

                continue



            # determine the file extension of the current file

            ext = filename[filename.rfind("."):].lower()



            # check to see if the file is an image and should be processed

            if validExts is None or ext.endswith(validExts):

                # construct the path to the image and yield it

                imagePath = os.path.join(rootDir, filename)

                yield imagePath



def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # initialize the dimensions of the image to be resized and

    # grab the image size

    dim = None

    (h, w) = image.shape[:2]



    # if both the width and height are None, then return the

    # original image

    if width is None and height is None:

        return image



    # check to see if the width is None

    if width is None:

        # calculate the ratio of the height and construct the

        # dimensions

        r = height / float(h)

        dim = (int(w * r), height)



    # otherwise, the height is None

    else:

        # calculate the ratio of the width and construct the

        # dimensions

        r = width / float(w)

        dim = (width, int(h * r))



    # resize the image

    resized = cv2.resize(image, dim, interpolation=inter)



    # return the resized image

    return resized
final_test_path = '../input/dogs-vs-cats-redux-kernels-edition/test/'

final_test_img_paths = list(list_images(final_test_path))

final_img_names=[i.split(os.path.sep)[-1].split('.jpg')[0] for i in final_test_img_paths]
len(final_test_img_paths)
# /kaggle/input/resnet50-imagenet-weights-dogs-vs-cats/resnet_dogs_vs_cats_testset_features.hdf5

# print(features.shape, labels.shape)
db_test = '../input/resnet50-imagenet-weights-dogs-vs-cats/resnet_dogs_vs_cats_testset_features.hdf5'

db_test = h5py.File(db_test, 'r')

features_test = db_test.get('features').value

#labels =  db_test.get('labels').value#.reshape(25000,-1)

print(features_test.shape)#, labels.shape)
predictions = model_logreg.predict_proba(features_test)

print(predictions.shape)
predictions
prediction_dog = predictions[:,1]

prediction_dog
# for dirname, _, filenames in os.walk(final_test_path):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
submission = pd.DataFrame({'id':final_img_names, 'label':prediction_dog})

submission.sort_values(by='id', ascending=True, inplace=True)

submission.head()


submission.to_csv('submission.csv',index=False)
db_test.close()

db.close()
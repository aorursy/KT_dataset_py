# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import cv2

from glob import glob

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

image_path = os.path.join("../input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png")

mask_path = os.path.join("../input/chest-xray-masks-and-labels/data/Lung Segmentation/masks")

reading_path = os.path.join("../input/chest-xray-masks-and-labels/data/Lung Segmentation/ClinicalReadings")



# Any results you write to the current directory are saved as output.
images = os.listdir(image_path)

masks = os.listdir(mask_path)

readings = os.listdir(reading_path)

print('Total number of x-ray images:',len(images))

print('Total number of masks:',len(masks))

print('Total number of clinical readings:',len(readings))
tb_positive = [image for image in images if image.split('.png')[0][-1]=='1']

tb_negative = [image for image in images if image.split('.png')[0][-1]=='0']

print('There are %d tuberculosis positive cases.' % len(tb_positive))

print('There are %d tuberculosis negative cases.' % len(tb_negative))
from IPython.display import Image

pos_image = np.random.choice(tb_positive,1)

neg_image = np.random.choice(tb_negative,1)

print("Image %s is positive on tuberculosis." % pos_image[0])

display(Image(os.path.join("../input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png",pos_image[0]),width=256,height=256))

print("Image %s is negative on tuberculosis." % neg_image[0])

display(Image(os.path.join("../input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png",neg_image[0]),width=256,height=256))
tb_state = [int(image.split('.png')[0][-1]) for image in images]

img_df = pd.DataFrame({'Image_name':images, 'TB_state': tb_state})

img_df['Path'] = img_df['Image_name'].map(lambda x: "../input/chest-xray-masks-and-labels/data/Lung Segmentation/CXR_png/"+x)

img_df['Source'] = img_df['Image_name'].map(lambda x: x.split('_')[0])

img_df['Text_path'] = img_df['Image_name'].map(lambda x: "../input/chest-xray-masks-and-labels/data/Lung Segmentation/ClinicalReadings/"+x.split('.png')[0]+'.txt')

img_df.head()
ages=[]

genders=[]

descriptions=[]

for txt in img_df.Text_path.tolist():

    lines = [line for line in open(txt,'r')]

    if "Patient's Sex:" in lines[0]:

        gender = lines[0][lines[0].index("Patient's Sex:")+len("Patient's Sex:")+1]

        genders.append(gender)

        start = lines[1].index("Patient's Age:")

        length = len("Patient's Age:")

        age = int(lines[1][start+length+2:start+length+4])

        ages.append(age)

        description = ' '.join(lines[2:]).strip()

        descriptions.append(description)

    else:

        if "male" or "MALE" in lines[0]:

            gender = 'M'

            genders.append(gender)

        else:

            gender = 'F'

            genders.append(gender)

        if "yrs" in lines[0]:

            start = lines[0].index("yrs")

            age = int(lines[0][start-2:start])

            ages.append(age)

        elif "yr" in lines[0]:

            start = lines[0].index("yr")

            age = int(lines[0][start-2:start])

            ages.append(age)

        else:

            ages.append(np.NaN)

        description = ' '.join(lines[1:]).strip()

        descriptions.append(description)

            

img_df['Age'] = ages

img_df['Gender'] = genders

img_df['Description'] = descriptions

img_df.head()
sns.countplot(x='TB_state', data=img_df)
img_df.groupby(by='Source')['Image_name'].count()
sns.countplot(x='Source', data=img_df)
sns.countplot(x='Gender', hue='TB_state', data=img_df)
sum((img_df.Gender=='M'))
sum((img_df.Gender=='F'))
img_df[img_df.Gender=='O']
print('TB positive rate of male patients:',sum((img_df.Gender=='M') & (img_df.TB_state==1)) / sum(img_df.Gender=='M'))
print('TB positive rate of female patients:',sum((img_df.Gender=='F') & (img_df.TB_state==1)) / sum(img_df.Gender=='F'))
img_df[img_df.Age.isnull()]
null_age_imgs = img_df[img_df.Age.isnull()].Text_path

for txt in null_age_imgs:

    lines = [line for line in open(txt,'r')]

    print(lines)
img_df.ix[446,'Age']=1

img_df.ix[469,'Age']=0

img_df.ix[535,'Age']=1

img_df.ix[660,'Age']=42

img_df[img_df.Age.isnull()]
sns.distplot(img_df[img_df.TB_state==1]['Age'], kde=False)
import time

start = time.time()



import numpy as np

from tensorflow.python.keras.applications import ResNet50, InceptionV3, InceptionResNetV2

from tensorflow.python.keras.preprocessing import image

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.applications.resnet50 import preprocess_input

#from tensorflow.python.keras.applications.inception_v3 import preprocess_input

#from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input



resnet_weights_path = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

#inceptionv3_weights_path = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

#inceptionresnetv2_weights_path = '../input/keras-pretrained-models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'



base_model = ResNet50(weights=resnet_weights_path)

model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)



image_size = 224

img_paths = img_df.Path.tolist()

features_array = np.zeros((800,2048))



for i, img_path in enumerate(img_paths):

    img = image.load_img(img_path, target_size=(image_size, image_size))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    features = model.predict(x)

    features = features.reshape(2048,)

    features_array[i,:] = features

    

print('Running time: %.4f seconds' % (time.time()-start))
model.summary()
features_array.shape
img_df.head(2)
df = pd.DataFrame(features_array)

df['Image_name'] = img_df.Image_name

df['TB_state'] = img_df.TB_state

df.head(2)
from sklearn.model_selection import train_test_split

X = df.drop(['Image_name', 'TB_state'], axis=1)

y = df.TB_state

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
import time

start = time.time()



from sklearn.svm import SVC

clf = SVC()

clf.fit(Xtrain, ytrain)

preds = clf.predict(Xtest)



print('Running time: %.4f seconds' % (time.time()-start))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(ytest, preds))
cm = confusion_matrix(ytest, preds)

sns.heatmap(cm, annot=True, cbar=False)

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.title('Confusion Matrix')
from sklearn.metrics import classification_report

print(classification_report(ytest, preds))
from sklearn.model_selection import GridSearchCV
import time

start = time.time()



from sklearn.metrics import make_scorer

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}



clf_cv = GridSearchCV(SVC(), cv=5, param_grid={'C':[1,10,100,1000], 'gamma':[0.0001,0.001,0.01,0.1,1]}, verbose=1,

                      scoring=scoring, refit='AUC', return_train_score=True)

clf_cv.fit(Xtrain, ytrain)



print('Running time: %.4f seconds' % (time.time()-start))
best_clf_cv = clf_cv.best_estimator_

best_clf_cv
clf_cv.best_score_
preds_cv = best_clf_cv.predict(Xtest)

print(accuracy_score(ytest, preds_cv))
cm = confusion_matrix(ytest, preds_cv)

sns.heatmap(cm, annot=True, cbar=False)

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.title('Confusion Matrix')
from sklearn.metrics import classification_report

print(classification_report(ytest, preds_cv))
norms = np.linalg.norm(features_array, axis = 1, keepdims=True)

X_norm = features_array/norms

X_norm.shape
Xtrain2, Xtest2, ytrain2, ytest2 = train_test_split(X_norm, y, test_size=0.2)
start = time.time()



from sklearn.metrics import make_scorer

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}



clf_cv_norm = GridSearchCV(SVC(), cv=5, param_grid={'C':[1,10,100,1000], 'gamma':[0.0001,0.001,0.01,0.1,1]}, verbose=1,

                      scoring=scoring, refit='AUC', return_train_score=True)

clf_cv_norm.fit(Xtrain2, ytrain2)



print('Running time: %.4f seconds' % (time.time()-start))
best_clf_cv_norm = clf_cv_norm.best_estimator_

best_clf_cv_norm
clf_cv_norm.best_score_
preds_cv_norm = best_clf_cv_norm.predict(Xtest2)

print(accuracy_score(ytest2, preds_cv_norm))
cm = confusion_matrix(ytest2, preds_cv_norm)

sns.heatmap(cm, annot=True, cbar=False)

plt.xlabel('Predicted label')

plt.ylabel('True label')

plt.title('Confusion Matrix')
from sklearn.metrics import classification_report

print(classification_report(ytest2, preds_cv_norm))
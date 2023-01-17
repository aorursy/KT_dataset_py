# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

from glob import iglob, glob

import matplotlib.pyplot as plt

from itertools import chain



%matplotlib inline
dataframe = pd.read_csv("/kaggle/input/sample/sample_labels.csv")

all_image_paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('/kaggle/input/sample/*','images*', '*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', dataframe.shape[0])

dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)

dataframe['Patient Age'] = dataframe['Patient Age'].map(lambda x: int(x[:-1]))
dataframe = dataframe[dataframe['Finding Labels'] != 'No Finding']

all_labels = np.unique(list(chain(*dataframe['Finding Labels'].map(lambda x: x.split('|')).tolist())))

pathology_list = all_labels

dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)

dataframe = dataframe.drop(['Patient Age', 'Patient Gender', 'Follow-up #', 'Patient ID', 'View Position', 

         'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacing_x','OriginalImagePixelSpacing_y'], axis=1)

for pathology in pathology_list :

    dataframe[pathology] = dataframe['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)

dataframe = dataframe.drop(['Image Index', 'Finding Labels'], axis=1)
dataframe.head()
dataframe['disease_vec'] = dataframe.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
dataframe.head()
from sklearn.model_selection import train_test_split



train_df, test_df = train_test_split(dataframe, 

                                   test_size = 0.25, 

                                   random_state = 2018)
X_train = train_df['path'].values.tolist()

y_train = np.asarray(train_df['disease_vec'].values.tolist())

X_test = test_df['path'].values.tolist()

y_test = np.asarray(test_df['disease_vec'].values.tolist())
print(X_train[0],y_train[0])
X_test[0]
from skimage.io import imread, imshow

print(imread(X_train[0]).shape)

images_train = np.zeros([len(X_train),128,128])

for i, x in enumerate(X_train):

    image = imread(x, as_gray=True)[::8,::8]

    images_train[i] = (image - image.min())/(image.max() - image.min())

images_test = np.zeros([len(X_test),128,128])

for i, x in enumerate(X_test):

    image = imread(x, as_gray=True)[::8,::8]

    images_test[i] = (image - image.min())/(image.max() - image.min())
X_train = images_train.reshape(len(X_train), 128, 128, 1)

X_test = images_test.reshape(len(X_test), 128, 128, 1)

X_train.astype('float32')
X_test[0].shape
from keras.applications.nasnet import NASNetLarge
from keras.models import Sequential

from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.applications.resnet import ResNet50

base_model1 = NASNetLarge(input_shape = (128, 128, 1), 

                                 include_top = False, weights = None)

model1 = Sequential()

model1.add(base_model1)

model1.add(GlobalAveragePooling2D())

model1.add(Dropout(0.3))

model1.add(Dense(512))

model1.add(Dropout(0.3))

model1.add(Dense(len(all_labels), activation='softmax'))

model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()
history1 = model1.fit(X_train, y_train, epochs = 30, verbose=1, validation_data=(X_test, y_test))
from keras.applications.densenet import DenseNet121
from keras.models import Sequential

from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.applications.xception import Xception

base_model = DenseNet121(input_shape = (128, 128, 1), 

                                 include_top = False, weights = None)

model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.3))

model.add(Dense(512))

model.add(Dropout(0.3))

model.add(Dense(len(all_labels), activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, epochs = 30, verbose=1, validation_data=(X_test, y_test))
def history_plot(history):

    plt.plot(history.history['acc'])

    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
predictions1 = model1.predict(X_test, batch_size = 32, verbose = True)
predictions = model.predict(X_test, batch_size = 32, verbose = True)
predictions
y_test.astype()
from sklearn.metrics import multilabel_confusion_matrix

class_p = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']



multilabel_confusion_matrix(predictions,y_test,class_p)
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix



fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_labels):

    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), predictions[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
from sklearn.metrics import roc_curve, auc

fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

for (idx, c_label) in enumerate(all_labels):

    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), predictions1[:,idx])

    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))

c_ax.legend()

c_ax.set_xlabel('False Positive Rate')

c_ax.set_ylabel('True Positive Rate')

fig.savefig('barely_trained_net.png')
sickest_idx = np.argsort(np.sum(y_test, 1)<1)

fig, m_axs = plt.subplots(4, 4, figsize = (16, 32))

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):

    c_ax.imshow(X_test[idx, :,:,0], cmap = 'bone')

    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, y_test[idx]) if n_score>0.5]

    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, y_test[idx], predictions[idx]) if (n_score>0.5) or (p_score>0.5)]

    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png')
sickest_idx = np.argsort(np.sum(y_test, 1)<1)

fig, m_axs = plt.subplots(4, 4, figsize = (16, 32))

for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):

    c_ax.imshow(X_test[idx, :,:,0], cmap = 'bone')

    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, y_test[idx]) if n_score>0.5]

    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, y_test[idx], predictions1[idx]) if (n_score>0.5) or (p_score>0.5)]

    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))

    c_ax.axis('off')

fig.savefig('trained_img_predictions.png')
model.save('chest-xray.h5')
from sklearn.externals import joblib

joblib.dump(model, 'model.pkl')

print("Model dumped!")
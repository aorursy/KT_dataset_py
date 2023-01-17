# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir('.'))

# Any results you write to the current directory are saved as output.
import numpy as np
from numpy.random import seed
seed(2018)
from tensorflow import set_random_seed
set_random_seed(2018)
import pandas as pd

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, TimeDistributed
from keras.optimizers import Adam,Adadelta,Nadam,Adamax,RMSprop,SGD
from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Conv2D, AveragePooling2D
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Flatten
def get_model(timeseries, nfeatures, nclass):
    
    model = Sequential()
    model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35, return_sequences=True, input_shape=(timeseries, nfeatures)))
    model.add(LSTM(units=64, dropout=0.05, recurrent_dropout=0.35, return_sequences=True))
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, return_sequences=False))
    model.add(Dense(units=nclass, activation='softmax'))
    
    return model
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def balance_class_by_under_sampling(X, y):
    Xidx = [[xidx] for xidx in range(len(X))]
    y_cls = [np.argmax(one) for one in y]
    classset = sorted(list(set(y_cls)))
    sample_distribution = [len([one for one in y_cls if one == cur_cls]) for cur_cls in classset]
    nsamples = np.max(sample_distribution)
    flat_ratio = {cls:nsamples for cls in classset}
    print(flat_ratio)
    #print(Xidx.shape)
    #print(y_cls.shape)
    Xidx_resampled, y_cls_resampled = RandomUnderSampler(random_state=42).fit_sample(Xidx, y_cls)
    #print(Xidx_resampled.shape)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    #print(sampled_index.shape)
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

def balance_class_by_over_sampling(X, y):
    Xidx = [[xidx] for xidx in range(len(X))]
    y_cls = [np.argmax(one) for one in y]
    classset = sorted(list(set(y_cls)))
    sample_distribution = [len([one for one in y_cls if one == cur_cls]) for cur_cls in classset]
    nsamples = np.max(sample_distribution)
    flat_ratio = {cls:nsamples for cls in classset}
    print(flat_ratio)
    Xidx_resampled, y_cls_resampled = RandomOverSampler(ratio=flat_ratio, random_state=42).fit_sample(Xidx, y_cls)
    #print(Xidx_resampled.shape)
    sampled_index = [idx[0] for idx in Xidx_resampled]
    #print(sampled_index.shape)
    return np.array([X[idx] for idx in sampled_index]), np.array([y[idx] for idx in sampled_index])

data = np.load('../input/train.npz')
X, gender, region = data['X'], data['gender'], data['region']

X,region = balance_class_by_under_sampling(X, region)

print('Original X shape: {}'.format(X.shape))
#print(X[0][1])
# X = X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
# print('Reshape: {}'.format(X.shape))

X_train, X_test, region_train, region_test = train_test_split(X, region, test_size=0.2, random_state=2018)

publictest = np.load('../input/publictest.npz')
X_publictest, fname = publictest['X'], publictest['name']
# from imblearn.over_sampling import SMOTE 
# from collections import Counter

# y_ints = [y.argmax() for y in region_train]

# print('Original dataset shape {}'.format(Counter(y_ints)))

# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_sample(X_train, y_ints)
# print('Resampled dataset shape {}'.format(Counter(y_res)))
from sklearn.utils import class_weight
y_ints = [y.argmax() for y in region_train]
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_ints),y_ints)
class_weight_dict = dict(enumerate(class_weight))
print(class_weight_dict)
from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# from sklearn.svm import SVC

# svm = SVC(kernel='rbf')

# svm.fit(X_res,y_res)

# print('fitting done !!!')
# y_test= [y.argmax() for y in region_test]
# y_test = np.array(y_test)
# X_publictest = X_publictest.reshape((X_publictest.shape[0],X_publictest.shape[1]*X_publictest.shape[2]))
# print(X_publictest[1])
# print(y_test)

# from sklearn.decomposition import PCA

# # Make an instance of the Model
# pca = PCA(.95)
# pca.fit(X_res)
# print(pca.explained_variance_ratio_)
# print("Shape: ", pca.explained_variance_ratio_.shape)
# X_res = pca.transform(X_res)
# X_test = pca.transform(X_test)
# X_publictest = pca.transform(X_publictest)
# y_test=y_test.reshape(y_test.shape[0],1,1)
# print(y_test.shape)
# from sklearn import svm

# eval_set = [(X_test, y_test)]
# model = svm.LinearSVC(verbose =1).fit(X_res, y_res)

# print('fitting done !!!')
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split

# eval_set = [(X_test, y_test)]
# model = XGBClassifier(max_depth=5)
# model.fit(X_res, y_res, eval_metric="merror", eval_set=eval_set, verbose=True)

# from sklearn.metrics import accuracy_score
# y_pred = model.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# import itertools

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=0)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         #print("Normalized confusion matrix")
#     else:
#         1#print('Confusion matrix, without normalization')

#     #print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
# Create paths for model
# import os
# cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)
# models_dir = os.path.join(cache_dir, 'models')
# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)
# Copy model over
# !cp -r ../input/zalo-voice/ ../working
# Check that model is in place
# !ls ~/.keras/models/zalo-voice
# y_test = y_test.reshape(y_test.shape[0],1)
# print(X_res[0])
# print(y_res[0])
# print(X_test.shape)
# print(y_test.shape)
opt = Adam()
model = get_model(X_train.shape[1], X_train.shape[2], 3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[f1,'accuracy'])
model.summary()

batch_size = 1024
nb_epochs = 1000

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#model.load_weights(filepath)
model.fit(X_train, region_train, batch_size=batch_size,epochs=nb_epochs, validation_data=(X_test, region_test), callbacks=callbacks_list, verbose=2)
predicts = model.predict(X_publictest, batch_size = batch_size)
predicts = np.argmax(predicts, axis=1)

region_dict = {0:'north', 1:'central', 2:'south'}
gender_dict = {0:'female', 1:'male'}
for i in range(32):
    print(fname[i], '-->', region_dict[predicts[i]])

submit = pd.DataFrame.from_dict({'id':fname, 'accent':predicts}) 
submit.to_csv('RUS_region.csv', index=False)
submit

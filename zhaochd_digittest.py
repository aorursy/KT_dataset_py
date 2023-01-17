# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from xgboost.sklearn import XGBClassifier


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import np_utils

model = Sequential();
model.add(Convolution2D(4,5,5,border_mode='valid',input_shape=(1,28,28)));
model.add(Activation('tanh'));

model.add(Convolution2D(8,3,3,border_mode='valid'));
model.add(Activation('tanh'));
model.add(MaxPooling2D(pool_size=(2,2)));
model.add(Dropout(0.25))

model.add(Convolution2D(16,3,3,border_mode='valid'));
model.add(Activation('tanh'));
model.add(MaxPooling2D(pool_size=(2,2)));
model.add(Dropout(0.25))

model.add(Flatten());
model.add(Dense(128,init='glorot_uniform'));
model.add(Activation('tanh'));
model.add(Dropout(0.25))

model.add(Dense(10,init='glorot_uniform'));
model.add(Activation('softmax'));

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True) # 设定学习率（lr）等参数  
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical') # 使用交叉熵作为loss函数

np_class = 10;

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

trainlabel = train['label'].values;
train = train.drop(['label'],axis = 1).values;
test = test.values;
"""
#model = GradientBoostingClassifier(n_estimators=50);
model = XGBClassifier(max_depth=6,learning_rate=0.5,n_estimators=50,objective='multi:softprob',subsample=0.6,colsample_bytree=0.6,seed=0);
model = model.fit(train,trainlabel);
#训练误差等分析
print(model);
predicted = model.predict(train);
print(metrics.classification_report(trainlabel, predicted));
print(metrics.confusion_matrix(trainlabel, predicted));




test_predict = model.predict(test);
"""
train = train.reshape(train.shape[0],1,28,28);
test = test.reshape(test.shape[0],1,28,28);
y_train = np_utils.to_categorical(trainlabel,np_class);

model.fit(train,y_train,batch_size=100, nb_epoch=1, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.3)  
test_classes = model.predict_classes(test, batch_size=100)

print (test_classes)
"""
test_predict = pd.Series(test_predict);
imageid = range(1,test.shape[0]+1);
imageid = pd.Series(imageid);
test_df = pd.DataFrame({'ImageId':imageid,'Label':test_predict});
test_df.to_csv('digit.csv',index=False);
"""

print( test_classes[0:5])
test_predict = pd.Series(test_classes);
imageid = range(1,test.shape[0]+1);
imageid = pd.Series(imageid);
test_df = pd.DataFrame({'ImageId':imageid,'Label':test_predict});
test_df.to_csv('digit.csv',index=False);

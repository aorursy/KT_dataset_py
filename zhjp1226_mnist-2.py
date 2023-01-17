# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
data=pd.read_csv("../input/train.csv")
y=data.ix[:,0].values
X=data.ix[:,1:data.shape[1]].values
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
print(X_train.shape)
print(X_test.shape)
X_train_ = X_train.reshape(X_train.shape[0], 28, 28)

for i in range(0, 3):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train_[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]
gen=image.ImageDataGenerator()
batches=gen.flow(X_train,y_train,batch_size=64)
mean=np.mean(X_train)
std=np.std(X_train)

def standardize(x):
    return (x-mean)/std
#写一个LossHistory类，保存loss和acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()
    def acc_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc')
        plt.legend(loc="upper right")
        plt.show()
#创建一个实例history
history = LossHistory()
# from keras.callbacks import ReduceLROnPlateau
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=2, mode='auto',min_lr=0)
import keras.backend as K
from keras.callbacks import LearningRateScheduler
def model():
    model=Sequential()
    model.add(Lambda(standardize,input_shape=(28,28,1)))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),activation="relu"))
    model.add(Conv2D(128,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),activation="relu"))
    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
    def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/10
        if epoch % 20 == 0 and epoch != 0:
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    model.fit_generator(generator=batches,steps_per_epoch=batches.n/batches.batch_size,epochs=60,
                        validation_data=(X_test, y_test),callbacks=[history,reduce_lr])
    return model
model=model()
score=model.evaluate(X_test,y_test,verbose=0)
print("CNN Error:%.2f%%" %(100-score[1]*100))
X_test=pd.read_csv('../input/test.csv')
X_test=X_test.values.reshape(X_test.shape[0],28,28,1)
preds=model.predict_classes(X_test,verbose=1)
model.save('digit_recognizer.h5')
def write_preds(preds,fname):
    pd.DataFrame({"ImageId":list(range(1,len(preds)+1)),"Label":preds}).to_csv(fname,index=False,header=True)
write_preds(preds,"result.csv")
#绘制acc-loss曲线
history.loss_plot('epoch')
history.acc_plot('epoch')
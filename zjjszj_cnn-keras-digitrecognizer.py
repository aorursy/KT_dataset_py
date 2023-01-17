"""

1 未添加dropout时迭代10次，精度能达到0.99，添加了dropout，训练精度下降了，为0.96，因此添加了数据增强,

2 

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,add,Flatten,BatchNormalization,Activation,Dense,Dropout

from keras import optimizers,initializers,regularizers

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator   #数据增强

from sklearn.metrics import confusion_matrix

import itertools

#获取数据

train=pd.read_csv("../input/train.csv");

test=pd.read_csv("../input/test.csv");

x_train=train.drop(labels=["label"],axis=1);

y_train=train["label"]

import seaborn as sns

#打印数据直方图

g=sns.countplot(y_train)

#正则化数据

x_train=x_train/255.0

test=test/255.0

#将数据化为3D

x_train=x_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

y_train = to_categorical(y_train, num_classes = 10)

#将训练数据集拆分出验证集

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.1,random_state=2)

#画一个训练集的例子

g=plt.imshow(x_train[0][:,:,0])


"""

(conv--bn--relu--conv--bn--relu--pool--dropout)*2--

(conv--bn--relu--dropout)*2--pool--

fc--output

"""

model=Sequential()

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same",input_shape = (28,28,1),

                 kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))

#reply

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same",input_shape = (28,28,1),

                 kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Conv2D(filters=64,kernel_size=(5,5),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(rate=0.25))



model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Dropout(rate=0.5))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

#reply

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

model.add(BatchNormalization(momentum=0.9))

model.add(Activation('relu'))

model.add(Dropout(rate=0.25))

model.add(MaxPooling2D(pool_size=(2,2)))

#fc

model.add(BatchNormalization(momentum=0.9))

model.add(Flatten())#将输入展平

model.add(Dense(256,activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))

#output

model.add(Dense(10,activation='softmax', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)))



#设置模型

adma=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy', optimizer=adma,metrics=['accuracy'])


epochs=30

batch_size=86

history=model.fit(batch_size=batch_size,epochs=epochs,shuffle=True,x=x_train,y=y_train,validation_data=(x_val,y_val),verbose=2)

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

# Predict the values from the validation dataset

y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# 显示一些错误结果，及预测标签和真实标签之间的不同

errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]

y_pred_errors = y_pred[errors]

y_true_errors = y_true[errors]

x_val_errors = x_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)
# 对测试集做预测

results = model.predict(test)

# 把one-hot vector转换为数字

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

# 保存最终的结果

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_submission.csv",index=False)
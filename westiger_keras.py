#引入必要的包

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.preprocessing import image



from sklearn.model_selection import train_test_split
#读入数据

train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

#取前500条数据（节省时间）

train = train.head(500)

test = test.head(500)

#取出训练数据中标签

X_train = (train.ix[:,1:].values).astype('float32')

y_train = train.ix[:,0].values.astype('int32')



X_test = test.values.astype('float32')
#将图片转化为矩阵格式

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
#数据标准化

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)

def standardize(x): 

    return (x-mean_px)/std_px
#标签独热编码

from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

num_classes = y_train.shape[1]
#构建深度神经网络

model= Sequential()

model.add(Lambda(standardize,input_shape=(28,28,1)))

model.add(Flatten())



#ReLU (The Rectified Linear Unit) [f(x)=max(0,x)]

#model.add(Dense(128,activation='relu'))

#model.add(Dropout(0.15))



model.add(Dense(10, activation='softmax'))

print("input shape ",model.input_shape)

print("output shape ",model.output_shape)
#学习

model.compile(optimizer=RMSprop(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.optimizer.lr=0.01

gen = image.ImageDataGenerator()

batches = gen.flow(X_train, y_train, batch_size=64)

history=model.fit_generator(batches, batches.n, nb_epoch=1)
#预测

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})

submissions.to_csv("./prediction.csv", index=False, header=True)
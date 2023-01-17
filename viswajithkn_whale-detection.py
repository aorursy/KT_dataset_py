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

import matplotlib as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Dense,Activation,Dropout,Conv2D,MaxPooling2D,Flatten,Input,BatchNormalization,AveragePooling2D,LeakyReLU

from keras.metrics import top_k_categorical_accuracy

from keras.utils import plot_model

from keras import optimizers

from keras.models import Model

from tqdm import tqdm

from keras.applications.resnet50 import ResNet50
trainFilePath = '../input/train.csv'

rawTrainData = pd.read_csv(trainFilePath)

print(rawTrainData.head(15))
trainLabel = rawTrainData['Id']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(trainLabel)

trainLabel_transform = le.transform(trainLabel)

print('The number of unique whale classes are : ',len(np.unique(trainLabel_transform)))
import matplotlib.image as mpimg

import matplotlib.pyplot as plt

uniqueWhaleNames = np.unique(trainLabel)

tmpUniqueWhaleNames = uniqueWhaleNames[0:9]

for id in tmpUniqueWhaleNames:

    tempDF = rawTrainData.loc[rawTrainData['Id'] == id]

    fileName = tempDF.iloc[0]['Image']

    imgdata = mpimg.imread('../input/train/' + fileName)

    plt.imshow(imgdata)

    plt.show()

    print('The size of image is: ',np.shape(imgdata))
import cv2



train_df,val_df = train_test_split(rawTrainData,test_size = 0.03, random_state = 42)

batch_size = 64



def prepareImageData(fileName):    

    gray_image = cv2.imread('../input/train/' + fileName)    

    gray_image = cv2.resize(gray_image, (64, 64), interpolation = cv2.INTER_AREA)    

    return np.array(gray_image)        
def top_5_accuracy(y_true,y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=5)
new_whale_df = rawTrainData[rawTrainData.Id == "new_whale"] 

trainData = rawTrainData[~(rawTrainData.Id == "new_whale")] 
trainLabel = trainData['Id']

le = LabelEncoder()

le.fit(trainLabel)

trainLabel_transform = le.transform(trainLabel)

print('The number of unique whale classes are : ',len(np.unique(trainLabel_transform)))
train_df,val_df = train_test_split(trainData,test_size = 0.03, random_state = 42)

X_train = np.array([prepareImageData(fileName) for fileName in tqdm(train_df['Image'])])

Y_train = np.array([targetVal for targetVal in tqdm(train_df['Id'])])

Y_train = keras.utils.to_categorical(le.transform(Y_train),num_classes = len(np.unique(trainLabel_transform)))



X_val = np.array([prepareImageData(fileName) for fileName in tqdm(val_df['Image'])])

Y_val = np.array([targetVal for targetVal in tqdm(val_df['Id'])])

Y_val = keras.utils.to_categorical(le.transform(Y_val),num_classes = len(np.unique(trainLabel_transform)))
input_img = Input(shape=(64,64,3))

layer_1 = Conv2D(filters = 6,kernel_size = (5,5),strides = 1,padding = 'same')(input_img)

layer_1 = BatchNormalization(axis=3, momentum=0.99, epsilon=0.001)(layer_1)

layer_1 = Activation('relu')(layer_1)

layer_1 = MaxPooling2D(pool_size = (2,2),padding = 'same')(layer_1)

layer_2 = Conv2D(filters = 16,kernel_size = (5,5),strides = 1,padding = 'same')(layer_1)

layer_2 = Activation('relu')(layer_2)

layer_2 = MaxPooling2D(pool_size = (2,2),padding = 'same')(layer_2)

layer_3 = Conv2D(filters = 16,kernel_size = (5,5),strides = 1,padding = 'same')(layer_2)

layer_3 = Activation('relu')(layer_3)

layer_3 = MaxPooling2D(pool_size = (2,2),padding = 'same')(layer_3)

layer_4 = Flatten()(layer_3)

layer_4 = Dense(512,activation='relu')(layer_4)

layer_4 = Dropout(0.5)(layer_4)

output = Dense(len(np.unique(trainLabel_transform)),activation='softmax')(layer_4)

model_conv = Model(inputs = input_img, outputs = output)

model_conv.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['accuracy',top_5_accuracy])

model_conv.summary()
model_conv.fit(X_train, Y_train, batch_size=256,validation_data = (X_val,Y_val),epochs = 50)
X_newWhale = np.array([prepareImageData(fileName) for fileName in tqdm(new_whale_df['Image'])])
Y_newwhale_predict = model_conv.predict(X_newWhale, batch_size=256)
whaleThresh = np.mean(np.max(Y_newwhale_predict,axis=0))
def getLabel(classes,le):

    result = []

    _class = le.inverse_transform(classes)

    for i in range(0, len(classes)):              

        result.append(_class[i])

    return result
TEST = '../input/test/'

test_names = [f for f in os.listdir(TEST)]
SAMPLE_SUBMISSION_FILE="submission_64.csv"



with open(SAMPLE_SUBMISSION_FILE,"w") as f:

    f.write("Image,Id\n")

    for fileName in tqdm(test_names):        

        gray_image = cv2.imread('../input/test/' + fileName)

        gray_image = cv2.resize(gray_image,(64,64),interpolation = cv2.INTER_AREA)         

        X_test = np.array(np.reshape(gray_image,(1,64,64,3)))

        Y_test = model_conv.predict(X_test,batch_size=1)        

        temp_best_predict_5 = np.argsort(Y_test)[0][::-1][:5]            

        temp_pre = getLabel(temp_best_predict_5,le)

        best_Y_test = Y_test[0,temp_best_predict_5]        

        for i in range(0,len(best_Y_test)):

            print(best_Y_test[i])

            if best_Y_test[i] < whaleThresh:

                breakId = i

                break

        if breakId <4:

            pre = []

            for i in range(0,breakId):

                pre.append(temp_pre[i])

            pre.append("new_whale")        

            for i in range(breakId+1,4):

                pre.append(temp_pre[i])

        else:

            pre = temp_pre

            

        #print(image, " ".join( pre))

        print(pre)

        f.write("%s,%s\n" %(os.path.basename(fileName), " ".join( pre)))

print("csv created")
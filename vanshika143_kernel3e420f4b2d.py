import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import os

train_path = '/kaggle/input/facial-keypoints-detection/training.zip'

test_path = '/kaggle/input/facial-keypoints-detection/test.zip'



Id_table_path = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'



sample_sub_path = '/kaggle/input/facial-keypoints-detection/SampleSubmission.csv'



extracted_files_path = '/kaggle/working'
Id_table = pd.read_csv(Id_table_path)



sample_sub = pd.read_csv(sample_sub_path)
import zipfile

with zipfile.ZipFile(train_path, 'r') as zip_ref:

    zip_ref.extractall(extracted_files_path)

#Unzip test csv file to 'extracted_files_path'.

with zipfile.ZipFile(test_path, 'r') as zip_ref:

    zip_ref.extractall(extracted_files_path)
#Read train csv file.

train_csv = pd.read_csv(extracted_files_path + '/training.csv')

#Read test csv file.

test_csv = pd.read_csv(extracted_files_path + '/test.csv')

#Read IdLookUpTable csv file.

looktable_csv = pd.read_csv(Id_table_path)
train_csv.isnull().any().value_counts()
train_csv.fillna(method = 'ffill',inplace = True)
train_csv.isnull().any().value_counts()
train_csv['Image'] = train_csv['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' '))
Y = train_csv.drop("Image",axis = 1)



X_train = train_csv['Image']



X_train = X_train / 255



X = np.array([X_train], dtype=float)



X=X.reshape(X_train.shape[0],96,96,1)



Y



plt.imshow(X[7].reshape(96,96),cmap='gray')

plt.show()
from keras.models import Sequential



from keras.layers import Conv2D



from keras.layers import MaxPooling2D



from keras.layers import Dense,BatchNormalization,Flatten,Dropout



from keras.optimizers import Adam



import keras



from keras.layers.advanced_activations import LeakyReLU
model = Sequential()
keras.initializers.TruncatedNormal(mean=0,stddev = .05 )



model.add(Conv2D(filters =64,kernel_size = (3,3),input_shape = (96,96,1),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(Conv2D(filters =64,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size =(2,2),strides =(2,2),padding ="same"))



# layer 2



model.add(Conv2D(filters =128,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(Conv2D(filters =128,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size =(2,2),strides =(2,2),padding ="same"))



# layer 3



model.add(Conv2D(filters =256,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(Conv2D(filters =256,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size =(2,2),strides =(2,2),padding ="same"))



# layer 4



model.add(Conv2D(filters =512,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(Conv2D(filters =512,kernel_size = (3,3),strides =(1,1),padding ="same",bias_initializer ='zero'))



model.add(LeakyReLU(alpha=0.2))



model.add(BatchNormalization())



model.add(MaxPooling2D(pool_size =(2,2),strides =(2,2),padding ="same"))





# flatten



model.add(Flatten())



# fully connected layer 1



model.add(Dense(2048,activation ="relu"))



model.add(BatchNormalization())



model.add(Dropout(0.2))



# fully connected layer 2



model.add(Dense(512,activation ="relu"))



model.add(BatchNormalization())



model.add(Dropout(0.2))



model.add(Dense(30))



model.summary()

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=.999)
model.compile(optimizer = optimizer , loss = "mean_squared_error", metrics=["mae"])
model.fit(X,Y,epochs = 100,batch_size = 256,validation_split = 0.2)
test_data = test_csv['Image']



test_data1 = test_data.apply(lambda x: np.fromstring(x , dtype = int , sep =" "))



test_data1 /= 255



test_data1 = np.array([test_data1]).reshape((-1,96,96,1))

 

prediction = model.predict(test_data1)
Id_table.head(5)
sample_sub.head(5)
header = list(Y.columns)

data = pd.DataFrame(prediction ,columns =  header)



data.head(5)
for i in range(Id_table.shape[0]):

    Id_table.Location[i] = data.loc[Id_table.ImageId[i]-1][Id_table.FeatureName[i]]
sample_sub.Location = Id_table.Location
my_submission = sample_sub
my_submission.to_csv('SampleSubmission22.csv', index=False)
my_submission
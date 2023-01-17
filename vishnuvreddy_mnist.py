import keras
from keras.datasets import mnist
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA as pca
from sklearn.decomposition import TruncatedSVD as svd
from sklearn.manifold import TSNE as tsne
import warnings
warnings.filterwarnings(action='ignore')

from keras.layers import Conv2D,MaxPooling2D,Softmax,Dropout,Dense,Flatten
from keras.activations import relu 
from keras.models import Sequential
from keras import backend as K
#Load data into ipython notebook
train = pd.read_csv("../input/train.csv")
x_train = train[train.columns[1:]]
y_train = train[train.columns[0]]
x_test = pd.read_csv("../input/test.csv")
#Convert Data into 2D (28*28) such that data can pass into ConvNets 
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
x_train_2d = x_train.reshape(42000,28,28)
x_test_2d = x_test.reshape(28000,28,28)
#print size of train and test
print("Train Size is:",x_train.shape)
print("Test Size is:",x_test.shape)
#Lets visualize one data point
plt.imshow(x_train_2d[random.randrange(0,1000)],cmap='Greys')
y_train = y_train.reshape((42000,1))
pc = pca(n_components = 2)
pc_data = pc.fit_transform(x_train)
pc_data = pd.DataFrame(np.hstack((pc_data,y_train)),columns=['d1','d2','class'])
plt.figure(figsize=(10,9))
plt.title('PCA')
sns.scatterplot(x=pc_data['d1'],y=pc_data['d2'],hue=pc_data['class'],palette='Dark2',legend='full');
#Visualization with SVD
sv = svd(n_components = 2)
sv_data = sv.fit_transform(x_train)
sv_data = pd.DataFrame(np.hstack((sv_data,y_train)),columns=['d1','d2','class'])
plt.figure(figsize=(10,9))
plt.title('SVD')
sns.scatterplot(x=sv_data['d1'],y=sv_data['d2'],hue=sv_data['class'],palette='CMRmap',legend='full');
#Visualize MNIST with T-SNE
#Usually t-sne is time complex thing so we don't take complete dataset 

x_train_tsne = x_train[:20000]
y_train_tsne = y_train[:20000]
ts = tsne(n_components = 2)
tsne_data = ts.fit_transform(x_train_tsne)
tsne_data = pd.DataFrame(np.hstack((tsne_data,y_train_tsne)),columns=['d1','d2','class'])
plt.figure(figsize=(10,9))
plt.title('T-SNE')
sns.scatterplot(x=tsne_data['d1'],y=tsne_data['d2'],hue=tsne_data['class'],palette='CMRmap',legend='full');
#Prepare data to pass into ConvNets
img_rows,img_cols = 28,28
if K.image_data_format() == 'channels_first':
    x_train_2d = x_train_2d.reshape(x_train_2d.shape[0], 1, img_rows, img_cols)
    x_test_2d = x_test_2d.reshape(x_test_2d.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train_2d = x_train_2d.reshape(x_train_2d.shape[0], img_rows, img_cols, 1)
    x_test_2d = x_test_2d.reshape(x_test_2d.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
#Data Normalization 
x_train_2d = x_train_2d.astype('float32')
x_test_2d = x_test_2d.astype('float32')
x_train_2d /= 255
x_test_2d /= 255
print('x_train shape:', x_train.shape)
print(x_train_2d.shape[0], 'train samples')
print(x_test_2d.shape[0], 'test samples')

#convert y classes to binary classes
y_train = keras.utils.to_categorical(y_train)
#Build CNN architecture
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
#Model training
model.compile(optimizer='Adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(x=x_train_2d,y=y_train,batch_size=50,epochs=15,validation_split=0.6)
#predicting test data and store to a CSV file
result = pd.DataFrame(model.predict_classes(x_test_2d))
result.index += 1
result = result.reset_index()
result.columns = ['ImageId','Label']

#a.columns=['ImageId','Label']
result.to_csv("result.csv",index=False)
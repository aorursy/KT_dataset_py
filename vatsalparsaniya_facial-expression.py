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
import matplotlib.pyplot as plt
import seaborn as sns
raw_data = pd.read_csv("../input/fer2018/fer20131.csv")
raw_data.head()
raw_data.info()
emotions = {0:'Angry', 1:'Fear', 2:'Happy',3:'Sad', 4:'Surprise', 5:'Neutral'}
values_count = raw_data['Usage'].value_counts()
sizes =[values_count[0],values_count[1],values_count[-1]]
plt.figure(figsize=(7,7))
g = plt.pie(sizes,autopct='%1.1f%%',shadow=False,explode=(0,0.05,0.05),
           wedgeprops={"edgecolor":"k",'linewidth': 1, 'linestyle': 'dashed', 'antialiased': True},
           labels = ["Training = {}".format(values_count[0]),"Testing = {}".format(values_count[-1]),
                    "Devlopment = {}".format(values_count[1])])
plt.title("Data Distribution")
plt.legend(loc = 4)
plt.show()


trainig_data = raw_data[raw_data['Usage'] == 'Training'].iloc[:,:-1]
testing_data = raw_data[raw_data['Usage'] == 'PrivateTest'].iloc[:,:-1]
dev_data = raw_data[raw_data['Usage'] == 'PublicTest'].iloc[:,:-1]
def get_data(raw_data):
#     raw_data['pixels'] = raw_data['pixels'].apply(lambda x: np.array(x.split(),dtype=np.float32).reshape((48,48)))
    pixels = []
    target = []
    for x,y in zip(raw_data['pixels'],raw_data['emotion']):
        pixels.append(np.array(x.split(),dtype=np.float32))
        target.append(y)

    return np.array(pixels,dtype=np.float32).reshape((len(pixels),48,48,1)),np.array(target,dtype=np.float32).reshape((len(target),1))
X_train,y_train = get_data(trainig_data)
X_test,y_test = get_data(testing_data)
X_dev,y_dev = get_data(dev_data)
print("X_train shape : ",X_train.shape)
print("Y_train shape : ",y_train.shape)
print("X_dev shape : ",X_dev.shape)
print("Y_dev shape : ",y_dev.shape)
print("X_test shape : ",X_test.shape)
print("Y_test shape : ",y_test.shape)
def plot_distribution(y_train,y_test,y_dev):
    plt.figure(figsize=(10,10))
    plt.subplot(3,1,1)
    unique_data,frequency = np.unique(y_train,return_counts=True)
    sns.barplot(x=np.vectorize(emotions.get)(unique_data),y=frequency)
    
    plt.subplot(3,1,2)
    unique_data,frequency = np.unique(y_test,return_counts=True)
    sns.barplot(x=np.vectorize(emotions.get)(unique_data),y=frequency)
    
    plt.subplot(3,1,3)
    unique_data,frequency = np.unique(y_dev,return_counts=True)
    sns.barplot(x=np.vectorize(emotions.get)(unique_data),y=frequency)
    plt.show()
plot_distribution(y_train,y_test,y_dev)
from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=1)
tsne_data = model.fit_transform(X_train[0:1000].reshape(len(X_train[0:1000]),2304))
tsne_data_np = np.hstack((tsne_data,y_train[0:1000]))
tsne_dataframe = pd.DataFrame(data=tsne_data_np,columns=["Dim_1","Dim_2","label"])

plt.figure(figsize=(10,10))
sns.scatterplot(x="Dim_1",y="Dim_2",data=tsne_dataframe,hue='label', palette="Set3")
plt.title("TSNE")
plt.legend(title='emotions', loc='upper left', labels=['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral'])
plt.plot()
from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(X_train[0:1000].reshape(len(X_train[0:1000]),2304))
pca_data_np = np.hstack((pca_data,y_train[0:1000]))
pca_dataframe = pd.DataFrame(data=pca_data_np,columns=["1st_principal","2nd_principal","label"])

plt.figure(figsize=(10,10))
sns.scatterplot(x="1st_principal",y="2nd_principal",data=pca_dataframe,hue='label', palette="Set3")
plt.title("PCA")
plt.legend(title='emotions', loc='upper left', labels=['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral'])
plt.plot()
X_train = X_train/255.0
X_test = X_test/255.0
X_dev = X_dev/255.0

y_train = (np.arange(6) == y_train[:]).astype(np.float32)
y_test = (np.arange(6) == y_test[:]).astype(np.float32)
y_dev = (np.arange(6) == y_dev[:]).astype(np.float32)
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
def my_model():
    model = Sequential()
    input_shape = (48,48,1)
    
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(6, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy','mean_squared_error'],optimizer='adam')
    
    return model
model = my_model()
model.summary()
K.tensorflow_backend.clear_session()
path_model='model_filter.h5'
model=my_model() 
# K.set_value(model.optimizer.lr,1e-4) 
h=model.fit(x=X_train,y=y_train, 
            batch_size=128, 
            epochs=20, 
            verbose=1, 
            validation_data=(X_dev,y_dev),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )
plt.plot(h.history['loss'], label='MAE (testing data)')
plt.plot(h.history['val_loss'], label='MAE (validation data)')
plt.title('MAE')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
# Plot history: MSE
plt.plot(h.history['mean_squared_error'], label='MSE (testing data)')
plt.plot(h.history['val_mean_squared_error'], label='MSE (validation data)')
plt.title('MSE')
plt.ylabel('MSE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print ("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

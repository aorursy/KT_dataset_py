import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import os
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import cv2
import os
import glob
from tqdm import tqdm
def get_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (224, 224))
    return img

def readData(path,label=0):
    data_path = os.path.join(path, '*g')
    files = glob.glob(data_path)
    imgs=np.asarray(list(map(get_img,files)))
    labels=np.ones(len(files)) if label==1 else np.zeros(len(files))
    return imgs,labels
img_train_n = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL" # Enter Directory of all images 
img_test_n = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL" # Enter Directory of all images 
img_train_p = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA" # Enter Directory of all images 
img_test_p = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA" # Enter Directory of all images 
img_val_n = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL" # Enter Directory of all images 
img_val_p = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA" # Enter Directory of all images 

train_n,lable_train_n=readData(img_train_n,0)
test_n,lable_test_n=readData(img_test_n,0)
train_p,lable_train_p=readData(img_train_p,1)
test_p,lable_test_p=readData(img_test_p,1)
val_p,lable_val_p=readData(img_val_p,1)
val_n,lable_val_n=readData(img_val_n,0)

x_train = np.concatenate((train_n,train_p),0)
y_train = np.concatenate((lable_train_n,lable_train_p),0)
x_val = np.concatenate((val_n,val_p),0)
y_val = np.concatenate((lable_val_n,lable_val_p),0)
x_test = np.concatenate((test_n,test_p),0)
y_test = np.concatenate((lable_test_n,lable_test_p),0)

def plot_accuracy(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
class Net:
    def __init__(self,x_train,y_train,x_val,y_val,x_test,y_test,model=None):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test =x_test
        self.y_test =y_test
        self.x_val  =x_test
        self.y_val  =y_test
        self.model=model
        self.train=None
        
    def run(self,epochs = 100, batch_size=32, verbose=1):
        assert self.model is not None, "NO MODEL"
        self.model.summary()
        self.train = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)

    def evaluat(self,verbose=0):
        assert self.train is not None, "NOT TRAINED YET"
        # score
        self.scores = self.model.evaluate(self.x_test, self.y_test, verbose=verbose)
        print("Score : %.2f%%" % (self.scores[1]*100))
        # accuracy
        plot_accuracy(self.train)
        # cm
        y_pred = self.model.predict_classes(self.x_test)
        cm = confusion_matrix(y_pred,y_test)
        print(cm)
        # mispred pictures
        plt.figure(figsize=(15,25))
        n_test = x_test.shape[0]
        i=1
        for j in range(len(self.x_test)) :
            if (y_pred[j] != y_test[j]) & (i<50):
                plt.subplot(10,5,i)
                plt.axis('off')
                plt.imshow(np.squeeze(self.x_test[j]),cmap="gray")
                pred_classe = y_pred[j].argmax(axis=-1)
                plt.title('%d / %d' % (y_pred[j], y_test[j]))
                i+=1



size=(224,224,3)
#simple
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), input_shape=size, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dense(1,activation="sigmoid"))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Modèle CNN plus profond
model2 = Sequential()
model2.add(Conv2D(32, (3, 3), input_shape=size, activation='relu'))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))
model2.add(Conv2D(20, (3, 3), activation='relu'))

model2.add(Conv2D(20, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

model2.add(Conv2D(20, (3, 3), activation='relu'))
model2.add(Conv2D(20, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Dropout(0.2))

model2.add(Flatten())
model2.add(Dense(1,activation="sigmoid"))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.applications import VGG16
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=size)
vgg16.trainable = False
model3 = Sequential()
model3.add(vgg16)
model3.add(Flatten())
model3.add(Dense(512, activation='relu'))
model3.add(Dense(128, activation='relu'))
model3.add(Dense(1,activation="sigmoid"))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
models=[model1,model2,model3]
nets=map(lambda m:Net(x_train,y_train,x_val,y_val,x_test,y_test,m),models)
for n in nets:
    n.run()
    n.evaluat()


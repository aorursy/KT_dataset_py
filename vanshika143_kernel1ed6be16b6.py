import os # work with directory

import cv2

from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

img_size = 224

from keras.utils import to_categorical
print(os.listdir("/kaggle/input/belgiumts-dataset/BelgiumTSC_Training/Training"))

print(os.listdir("/kaggle/input/belgiumts-dataset/BelgiumTSC_Testing/Testing"))
def load_dataset(file_name):

    if file_name == "Training":

        directory = "/kaggle/input/belgiumts-dataset/BelgiumTSC_Training/Training"

    elif file_name =="Testing":

        directory = "/kaggle/input/belgiumts-dataset/BelgiumTSC_Testing/Testing"

        

    list_images =[]

    list_labels=[]

    

    count = 0

        

    for sub_dir in tqdm(os.listdir(directory)):



        

        if sub_dir == 'Readme.txt':

            pass

        

        else:

            inner_directory = os.path.join(directory,sub_dir)

            for image_file in os.listdir(inner_directory):

                

                if image_file.endswith(".ppm"):

                    

                    img = cv2.imread(os.path.join(inner_directory,image_file))

                    img = cv2.resize(img,(img_size,img_size))

                    

                    list_images.append(img)

                    

                    list_labels.append(count)

            count +=1   

            

    list_images = np.array(list_images).reshape(-1,img_size,img_size,3)        

                    

    list_labels = np.array(list_labels)

    list_labels= to_categorical(list_labels,count)        

  

      

    return list_images , list_labels              

                    

                    

                    

                    

            

            

            
training_dataset , training_label =load_dataset("Training")

testing_dataset , testing_label =load_dataset("Testing")

from sklearn.model_selection import train_test_split

X_train,X_val, Y_train, Y_val = train_test_split(training_dataset,training_label ,test_size=0.2, random_state=101,shuffle = True)
from keras.layers import Input ,Dense,Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

vgg = VGG16(input_shape = [img_size,img_size,3],weights='imagenet',include_top = False) # include_top = false means remove output layer

for layer in vgg.layers:

    layer.trainable = False
X = Flatten()(vgg.output)



layer1 = Dense(512,activation = "relu")(X)



layer2 = Dense(1536,activation = "relu")(layer1)



layer3 = Dense(3072,activation = "relu")(layer2)



layer4 = Dense(6144,activation = "relu")(layer3)



layer5 = Dense(12288,activation = "relu")(layer4)



prediction = Dense(Y_train.shape[1],activation = "softmax")(layer5) # output layer 
# create the model object



model = Model(inputs = vgg.input,outputs = prediction)
model.summary()
opt = Adam(learning_rate=0.0001)

model.compile(loss = "categorical_crossentropy",optimizer=opt,metrics =['accuracy'])
ans = model.fit(X_train ,Y_train , batch_size = 16,epochs =30,validation_data=(X_val,Y_val))
acc = ans.history['accuracy']

val_acc = ans.history['val_accuracy']

loss = ans.history['loss']

val_loss = ans.history['val_loss']

epochs = range(1,len(acc)+1)



plt.plot(epochs,acc,'bo',label ='Training_acc')

plt.plot(epochs,val_acc,'b',label ='validation_acc')

plt.title("training and validation accuracy")

plt.legend()



plt.figure()



plt.plot(epochs,loss,'bo',label ='Training_loss')

plt.plot(epochs,val_loss,'b',label ='validation_loss')

plt.title("training and validation accuracy")

plt.legend()



plt.show()
predict = np.argmax(model.predict(testing_dataset), axis = 1)

count = 0

for i in range(0,predict.shape[0]):

    if (predict[i] == np.argmax(testing_label[i])):

        count +=1

print ('Accuracy on Test ',100 * count/predict.shape[0],'%')
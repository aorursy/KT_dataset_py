import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
#loading dataset

path = '../input/Kannada-MNIST'

df = pd.read_csv(path+'/train.csv')

df_test = pd.read_csv(path+'/test.csv')

df.head()
#converting dataframe to 28x28 numpy array like an image



x_train = df.iloc[:,1:].values.reshape(df.shape[0],28,28)

x_test = df_test.iloc[:,1:].values.reshape(df_test.shape[0],28,28)



y_train = df.iloc[:,0].values

id_test = df_test.iloc[:,0].values



#Creating a stratified split to validate the model

x_train,x_val, y_train,y_val = train_test_split(x_train,y_train,stratify=y_train,test_size=0.1)
#distribution of categories

sns.countplot(y_train)
#printing a sample train image with label

sample_num = 3

plt.imshow(x_train[sample_num],cmap='gray')

print('label: '+str(y_train[sample_num]))
#normalizing the input data for better convergence

x_train = x_train/255

x_val = x_val/255

x_test = x_test/255
#Tensorflow Conv2D accepts input in the shape of [m,h,w,c] = [samples size, height, width, channels]

#Thus we need to add the channel size as 1 for grayscale image



x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
# making a CNN network using tf.keras



model = tf.keras.models.Sequential([

    

    #args for conv2 are self explanatory

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu', input_shape=(28, 28,1)),

    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),padding='same',activation='relu'),

    tf.keras.layers.Dropout(0.2),

    # 2nd layer

    tf.keras.layers.Conv2D(128,kernel_size=(3,3), strides=(1,1),padding='same', activation='relu'),

    tf.keras.layers.Conv2D(128,kernel_size=(3,3), strides=(2,2),padding='same', activation='relu'),

    tf.keras.layers.Dropout(0.2),

    # 3rd layer

    tf.keras.layers.Conv2D(256,kernel_size=(3,3), strides=(1,1),padding='same', activation='relu'),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    #FC hidden layer 1

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    #FC hidden layer 2

    tf.keras.layers.Dense(128, activation='relu'),    

    #since we have 10 classes we use 10 neurons with softmax for classification

    tf.keras.layers.Dense(10, activation='softmax')



])



#using adam (RMSProp+momentum) for fast convergence

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10)
plt.plot(history.history['accuracy'],label='train')

plt.plot(history.history['val_accuracy'],label='validation')

plt.ylabel('Accuracy')

plt.xlabel('Epochs')

plt.legend()
#finding the accuracy on validation set

val_loss,val_acc = model.evaluate(x_val,y_val)
predictions = model.predict(x_val)



#printing some of the predicitons for analysis

f,ax = plt.subplots(7,7,figsize=(15,15))



y_pred = np.argmax(predictions,axis=1)



for i in range(49):

    

    ax[int(i/7),int(i%7)].imshow(x_test[i,:,:,0],cmap='gray')

    ax[int(i/7),int(i%7)].set_title('Prediction:'+str(y_pred[i]))

    ax[int(i/7),int(i%7)].axis('off')

    

plt.tight_layout()

plt.show()
import seaborn as sns

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(15,10))

#Here 0 to 28 labels are mapped to their original categories

ax = sns.heatmap(confusion_matrix(y_val,y_pred),annot=True,cmap='GnBu');

ax.set_xlabel('Predicted values');

ax.set_ylabel('True values');

ax.set_title('Confusion matrix');
y_test_pred = np.argmax(model.predict(x_test),axis=1)



result = pd.DataFrame([id_test,y_test_pred],index=['id','label']).T

result = result.set_index('id')

result.to_csv('submission.csv')
#Saving the model weights to load later



model.save_weights("kannada_MNIST_model.h5")
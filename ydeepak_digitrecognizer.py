import numpy as np, pandas as pd

import random



import keras

from keras.models import Sequential

from keras.layers import Conv2D,Dense,Input, Dropout, Flatten, MaxPooling2D

from keras.optimizers import SGD, Adam

from keras.callbacks import LearningRateScheduler

from keras.utils import to_categorical



from sklearn.metrics import accuracy_score

from sklearn.model_selection import  train_test_split



import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams["figure.figsize"]= [5,10]

%%time

train = pd.read_csv('../input/train.csv', low_memory=True)

test = pd.read_csv('../input/test.csv', low_memory=True)
print(train.shape, test.shape) 

## here each image is 28x28 matrix, train contains an extra (first)column 'label' containing actual 

## decimal representation of the image.
train.columns
train.loc[:,'label'].value_counts()
rows = 2; columns = 5

fig = plt.figure(figsize=(14, 8))

for i in range(1, columns*rows+1):

    image = np.array(train.iloc[i,1:]).reshape((28,28))

    label = train.loc[i,'label']

    ax = fig.add_subplot(rows,columns,i)

    ax.imshow(image, cmap='gray')

    ax.set_title(label)    
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,1:], train.iloc[:,0], test_size=0.33, random_state=42, shuffle = True)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = np.array(X_train).reshape((X_train.shape[0],28,28,1)).astype('float32')

X_test = np.array(X_test).reshape((X_test.shape[0],28,28,1)).astype('float32')
## Normalization

X_train /= 255

X_test /= 255
print(y_train.value_counts(), y_test.value_counts())
y_train = to_categorical(y_train)

y_test = to_categorical(y_test) 
y_test.shape
def learning_rate_scheduler(epoch):

    if epoch <5:

        lr = 1e-3

    if epoch >= 5 and epoch <=20:  lr = 3e-4

    if epoch >20: lr= 1e-5

    return lr
model = Sequential()



model.add(Conv2D(input_shape = (28,28,1),data_format='channels_last', filters = 128, kernel_size=(5,5), activation ='relu'))

model.add(MaxPooling2D(3,3))

model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3),activation = 'relu'))

model.add(Conv2D(filters = 28, kernel_size = (3,3),activation = 'relu'))

model.add(MaxPooling2D(3,3))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer = Adam(lr = 0.01),loss = 'categorical_crossentropy', metrics = ['accuracy'])



epoch = 6

batch_size = 64

change_lr = LearningRateScheduler(learning_rate_scheduler)

history = model.fit(X_train, y_train, epochs=epoch, batch_size = batch_size, validation_data=[X_test,y_test], 

                    callbacks=[change_lr], shuffle = True)
loss = history.history['loss']

acc = history.history['acc']



val_acc = history.history['val_acc']

val_loss = history.history['val_loss']
row,col = (1,2)



fig = plt.figure(figsize = (10,5))



ax1 = fig.add_subplot(121)

ax1.plot(loss, color='red',label='train_loss')

ax1.set_title('loss')

ax1.plot(val_loss, color='green',label= 'val_loss')

ax1.legend(loc='upper right')



ax2 = fig.add_subplot(122)

ax2.plot(acc, color='red', label = 'train accuracy')

ax2.plot(val_acc, color='green', label='val_accuracy')

ax2.set_title('accuracy')

ax2.legend(loc='lower right')



plt.plot()

plt.tight_layout()
model.evaluate(X_test, y_test)
test = np.array(test)

test = test.reshape((test.shape[0],28,28,1))/255
out = model.predict(test)
i = random.randint(1,28000)

image = np.array(test[i]).reshape((28,28))

plt.imshow(image, cmap='gray')

plt.title(f'Predictions for index {i}: {np.argmax(out[i])}')
output = np.argmax(out,axis =1)
pd.DataFrame(output, columns =['output'])['output'].value_counts()

                                                      
ImageId = list(range(1,28001))
submission = pd.DataFrame({'ImageId':ImageId, 'Label':output})
submission.to_csv('submission.csv',index = 'False')
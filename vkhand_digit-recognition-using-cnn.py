import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import History 



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.sample(10)
test.sample(10)
train.shape
test.shape
train.isnull().any().describe()
test.isnull().any().describe()
Y_train = train["label"]
X_train = train.drop(labels="label",axis=1)
print (X_train.shape)
print (Y_train.shape)
X_train = X_train/255.0
test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#Confirming the X_train shape we earlier predicted
print (X_train.shape)

#confirming the test shape we earlier predicted
print(test.shape)
nrows = 2
ncols = 3
i = 0
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
for row in range(nrows):
    for col in range(ncols):
        ax[row,col].imshow(X_train[i][:,:,0])
        ax[row,col].set_title("True label :{}".format(Y_train[i]))
        i += 1

sns.countplot(Y_train)
Y_train.value_counts()
##### num_classes = 10 because we have 10 classes from 0 to 9
Y_train = to_categorical(Y_train, num_classes=10)
#also let's look at our modified Y_train for the  1st 6 images displayed above. Remember: index starts from 0
Y_train[0:6]
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation = "relu"))

model.add(Dense(128, activation = "relu"))

model.add(Dense(10, activation = "softmax"))

#Note: I didn't use any regularisation yet! let's see how well our model acts without regularisation like dropout! We can always iterate later :)
#For faster convergence, i've used 10 epochs. 20 epochs seems to work a bit better! Try changing it to 20 or even 30 for better accuracy
epochs = 10
batch_size = 100
optimizerSGD = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
optimizerAdam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)
optimizerRMSprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

history = History() #to keep track of accuracy parameters, we will see it's use soon
#training using SGD
model.compile(optimizer = optimizerSGD , loss = "categorical_crossentropy", metrics=["accuracy"])
historySGD = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
         validation_data = (X_val, Y_val), verbose = 2)
resultsSGD = model.predict(test)
#training using RMSprop
model.compile(optimizer = optimizerRMSprop , loss = "categorical_crossentropy", metrics=["accuracy"])
historyRMSprop = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
         validation_data = (X_val, Y_val), verbose = 2)
resultsRMSProp = model.predict(test)
#training using Adam
model.compile(optimizer = optimizerAdam , loss = "categorical_crossentropy", metrics=["accuracy"])
historyAdam = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 
         validation_data = (X_val, Y_val), verbose = 2)
resultsAdam = model.predict(test)
SGD_acc = historySGD.history['acc']
SGD_val_acc = historySGD.history['val_acc']
RMSprop_acc = historyRMSprop.history['acc']
RMSprop_val_acc = historyRMSprop.history['val_acc']
Adam_acc = historyAdam.history['acc']
Adam_val_acc = historyAdam.history['val_acc']
plt.plot(SGD_acc)
plt.plot(RMSprop_acc)
plt.plot(Adam_acc)
plt.legend(['SGD', 'RMSprop', 'Adam'], loc='lower right')
plt.title('Training accuracy: SGD vs RMSprop vs Adam')
plt.show()
plt.plot(SGD_val_acc)
plt.plot(RMSprop_val_acc)
plt.plot(Adam_val_acc)
plt.legend(['SGD', 'RMSprop', 'Adam'], loc='lower right')
plt.title('Validation accuracy: SGD vs RMSprop vs Adam')
plt.show()
fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(15, 5))
ax[0].plot(SGD_acc)
ax[0].plot(SGD_val_acc)
ax[0].legend(['SGD_train','SGD_val'], loc='lower right')
ax[0].set_title("SGD")

ax[1].plot(RMSprop_acc)
ax[1].plot(RMSprop_val_acc)
ax[1].legend(['RMSprop_train','RMSprop_val'], loc='lower right')
ax[1].set_title("RMSprop")

ax[2].plot(Adam_acc)
ax[2].plot(Adam_val_acc)
ax[2].legend(['Adam_train','Adam_val'], loc='lower right')
ax[2].set_title("Adam")
results = np.argmax(resultsAdam,axis=1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("submission.csv",index=False)

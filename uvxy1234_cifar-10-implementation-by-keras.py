import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle



import keras

from keras.preprocessing.image import ImageDataGenerator



from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten, BatchNormalization

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau



import os

print(os.listdir("../input"))

run_model1 = True
def load_cifar10_data(filename):

    with open('../input/'+ filename, 'rb') as file:

        batch = pickle.load(file, encoding='latin1')



    features = batch['data']

    labels = batch['labels']

    return features, labels
# Load files

batch_1, labels_1 = load_cifar10_data('data_batch_1')

batch_2, labels_2 = load_cifar10_data('data_batch_2')

batch_3, labels_3 = load_cifar10_data('data_batch_3')

batch_4, labels_4 = load_cifar10_data('data_batch_4')

batch_5, labels_5 = load_cifar10_data('data_batch_5')



test, label_test = load_cifar10_data('test_batch')
# Merge files

X_train = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)

Y_train = np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0)

dict_ = {0:'Airplane', 1:'Automobile', 2:'Bird', 3:'Cat', 4:'Deer',

         5:'Dog', 6:'Frog', 7:'Horse', 8:'Ship', 9:'Truck'}



def return_photo(batch_file):

    assert batch_file.shape[1] == 3072

    dim = np.sqrt(1024).astype(int)

    r = batch_file[:, 0:1024].reshape(batch_file.shape[0], dim, dim, 1)

    g = batch_file[:, 1024:2048].reshape(batch_file.shape[0], dim, dim, 1)

    b = batch_file[:, 2048:3072].reshape(batch_file.shape[0], dim, dim, 1)

    photo = np.concatenate([r,g,b], -1)

    return photo





X_train = return_photo(X_train)

X_test = return_photo(test)

Y_test = np.array(label_test)
def plot_image(number, file, label):

    fig = plt.figure(figsize = (3,2))

    #img = return_photo(batch_file)

    plt.imshow(file[number])

    plt.title(dict_[label[number]])

    

plot_image(12345, X_train, Y_train)
# The cifar-10 is designed to balance distribution that the counts for each classification are 5000

import seaborn as sns

sns.countplot(Y_train)

hist_Y_train = pd.Series(Y_train).groupby(Y_train).count()

print(hist_Y_train)
from sklearn import preprocessing

oh_encoder = preprocessing.OneHotEncoder(categories='auto')

oh_encoder.fit(Y_train.reshape(-1,1))

X_train_nor = X_train.astype('float32') / 255.0

X_test_nor = X_test.astype('float32') / 255.0

Y_train_oh = oh_encoder.transform(Y_train.reshape(-1,1)).toarray()

Y_test_oh = oh_encoder.transform(Y_test.reshape(-1,1)).toarray()
print('One-hot:')

print(Y_train_oh[:5])

print('\nLabel:')

print(Y_train[:5])
# Final check for dimensions before training

print('X_train shape:', X_train_nor.shape)

print('Y_train_oh shape:', Y_train_oh.shape)

print('X_test shape:', X_test_nor.shape)

print('Y_train_oh shape:', Y_test_oh.shape)
def build_basic_net(model, dropout=0.2):

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', strides=1,

                     input_shape=X_train.shape[1:], activation='relu'))

    model.add(Dropout(dropout))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', strides=1,

                     input_shape=X_train.shape[1:], activation='relu'))

    model.add(Dropout(dropout))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(10,activation='softmax'))
hist_dict = {}



if __name__ == '__main__' and run_model1:

    model = Sequential()

    build_basic_net(model)

    model.summary()

    

    adam = optimizers.Adam()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    hist_dict['run_model1'] = model.fit(X_train_nor, Y_train_oh, batch_size=64, epochs=10,

                                    shuffle=True, validation_split=0.2, verbose=1)
def model_predict(model, test_set):

    print("Generating test predictions...")

    predictions = model.predict_classes(test_set, verbose=1)

    print("OK.")

    return predictions





def write_preds(preds, filename):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(filename, index=False, header=True)
if __name__ == '__main__' and run_model1:

    predictions = model_predict(model, X_test_nor)

    print(predictions[:5])

    write_preds(predictions, "keras-model1.csv")

    

    scores = model.evaluate(X_test_nor, Y_test_oh)

    print("Accuracy=", scores[1])
# Plot the loss and accuracy curves for training and validation 

def loss_acc_plt(history):

    fig, ax = plt.subplots(2,1)

    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)

    



if run_model1:

    loss_acc_plt(hist_dict['run_model1'])

    

"""

    if run_model2:

    loss_acc_plt(hist_dict['run_model2'])

if run_model3:

    loss_acc_plt(hist_dict['run_model3'])

if run_model_adv:

    loss_acc_plt(hist_dict['run_model_adv'])

"""
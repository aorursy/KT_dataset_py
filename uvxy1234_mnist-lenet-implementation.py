import PIL

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras

from matplotlib import pyplot



from sklearn import preprocessing
# control loop

run_model1 = False

run_model2 = False

run_model3 = False

run_model_adv = True
#input data

train = pd.read_csv('../input/train.csv', delimiter=',')

test = pd.read_csv('../input/test.csv', delimiter=',') 
train.head()
train_size = train.shape[0]

test_size = test.shape[0]



X_train = train.iloc[:, 1:].values.astype('uint8')  #iloc[row, column]

Y_train = train.iloc[:, 0]

X_test = test.iloc[:, :].values.astype('uint8')



img_dimension = np.int32(np.sqrt(X_train.shape[1]))

img_rows, img_cols = img_dimension, img_dimension #28*28

nb_of_color_channels = 1



# Check Keras backend

if(keras.backend.image_dim_ordering()=="th"):

    # Reshape the data to be used by a Theano CNN. Shape is

    # (nb_of_samples, nb_of_color_channels, img_width, img_heigh)

    X_train = X_train.reshape(train.shape[0], nb_of_color_channels, img_rows, img_cols)

    X_test = X_test.reshape(test.shape[0], nb_of_color_channels, img_rows, img_cols)

    in_shape = (nb_of_color_channels, img_rows, img_cols)

else:

    # Reshape the data to be used by a Tensorflow CNN. Shape is

    # (nb_of_samples, img_width, img_heigh, nb_of_color_channels)

    X_train = X_train.reshape(train.shape[0], img_rows, img_cols, nb_of_color_channels)

    X_test = X_test.reshape(test.shape[0], img_rows, img_cols, nb_of_color_channels)

    in_shape = (img_rows, img_cols, nb_of_color_channels)





#X_train = X_train.reshape(train_size, img_dimension, -1)

#X_test = X_test.reshape(test_size, img_dimension, -1)



print('Data Information\n')

print('Training set size: {}\nTesting set size: {}'.format(train_size, test_size))

print('Image dimension: {0}*{0}'.format(img_dimension))





#free some memory space

#del train, test

# display some image

def display_digits(dim, X, Y_true, pred=None, random_seed=None):

    """ This function shows n images choiced randomly with their predicted(optional) and real labels

    dim: plots parameter, tuple with (nrows, ncols)

    

    """

    if random_seed is not None:

        np.random.seed(random_seed)

        

    nrows, ncols = dim

    indices = np.random.randint(Y_true.shape[0], size=dim)

    

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    plt.subplots_adjust(wspace=0.1, hspace=0.8)

    

    for row in range(nrows):

        for col in range(ncols):

            i = indices[row,col]

            ax[row,col].imshow(X[i,:,:,0], cmap='gray')

            if pred is not None:

                ax[row,col].set_title("id:{0}\nPredicted label:{1}\nTrue label:{2}".

                                      format(i,pred[i],Y_true[i]))

            else:

                ax[row,col].set_title("id:{0}\nTrue label:{1}".format(i,Y_true[i]))



                

display_digits(dim=(2,3), X=X_train, Y_true=Y_train)

#display_random_digits(dim=(2,3))
# display the distribution of labels

sns.countplot(Y_train)

hist_Y_train = Y_train.groupby(Y_train.values).count()

print(hist_Y_train)
# Normalization: Make the value floats in [0,1] instead of int in [0,255]

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train_nor = X_train / 255

X_test_nor= X_test / 255
oh_encoder = preprocessing.OneHotEncoder(categories='auto')

oh_encoder.fit(Y_train.values.reshape(-1,1))

Y_train_oh = oh_encoder.transform(Y_train.values.reshape(-1,1)).toarray()
print('One-hot:')

print(Y_train_oh[:5])

print('\nLabel:')

print(Y_train[:5])
# Just for record~ Another way for one-hot encoding (by keras)

from keras.utils.np_utils import to_categorical 

to_categorical(Y_train, Y_train.unique().shape[0])[:5]

# Final check for dimensions before training

print('X_train shape:', X_train.shape)

print('Y_train shape:', Y_train.shape)

print('X_test shape:', X_test.shape)
from keras.layers import Activation,Dropout,Dense,Conv2D,AveragePooling2D,Flatten,ZeroPadding2D,MaxPooling2D

from keras.models import Sequential

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau
def build_lenet5(model, input_shape=X_train.shape[1:], dropout=0):

    # N' = (N+2P-F)/S + 1

    S = [1,2,1,2,1]

    N_input = [28,28,14,10,5]

    P = [2,0,0,0,0]

    N = [28,14,10,5,1]

    F = [i[0] + 2*i[1] - i[3]*(i[2] - 1) for i in zip(N_input, P, N, S)] #[5,2,5,2]

    

    #Input: (28*28*1)

    #C1: (28*28*6)

    model.add(Conv2D(filters=6, kernel_size=(F[0],F[0]), padding='same', strides=S[0],

                     activation='relu', input_shape=input_shape))

    #S2: (14*14*6)

    model.add(MaxPooling2D(pool_size=F[1], strides=S[1]))

    #C3: (10*10*16)

    model.add(Conv2D(filters=16, kernel_size=(F[2],F[2]), padding='valid', strides=S[2],

                     activation='relu'))

    #S4: (5*5*16)

    model.add(MaxPooling2D(pool_size=F[3], strides=S[3]))

    #C5: (1*1*120)

    model.add(Conv2D(filters=120, kernel_size=(F[4],F[4]), padding='valid', strides=S[4],

                     activation='relu'))

    #New add: Dropout

    model.add(Dropout(dropout))

    model.add(Flatten()) #Same work as C5 the input image size is unchanged.

    

    #F6: (84)

    model.add(Dense(84, activation='relu'))

    #Output: (10)

    model.add(Dense(10, activation='softmax'))

    



if __name__ == '__main__' and run_model1:

    model = Sequential()

    build_lenet5(model, input_shape=X_train.shape[1:], dropout=0)

    model.summary()



hist_dict = {}



if __name__ == '__main__' and run_model1:

    adam = optimizers.Adam()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    hist_dict['run_model1'] = model.fit(X_train, Y_train_oh, batch_size=64, epochs=20,

                                    shuffle=True, validation_split=0.2, verbose=2)

    
def model_predict(model):

    print("Generating test predictions...")

    predictions = model.predict_classes(X_test, verbose=1)

    print("OK.")

    return predictions



def model_predict_val(model, set_check):

    print("Generating set predictions...")

    predictions = model.predict_classes(set_check, verbose=1)

    print("OK.")

    return predictions



def write_preds(preds, filename):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(filename, index=False, header=True)



if __name__ == '__main__' and run_model1:    

    predictions = model_predict(model)

    print(predictions[:5])

    write_preds(predictions, "keras-lenet5-basic.csv")





if __name__ == '__main__' and run_model2:

    model = Sequential()

    build_lenet5(model, input_shape=X_train.shape[1:], dropout=0.3)

    model.summary()

    

    adam = optimizers.Adam()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    hist_dict['run_model2'] = model.fit(X_train, Y_train_oh, batch_size=64, epochs=20, shuffle=True, validation_split=0.2, verbose=2)
if __name__ == '__main__' and run_model2:

    predictions = model_predict(model)

    print(predictions[:5])

    write_preds(predictions, "keras-lenet5-basic-droupout.csv")
# Ref: https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

# augmentation 各項具體效果 : https://zhuanlan.zhihu.com/p/30197320

"""

Augmentation:



Randomly rotate some training images by 10 degrees

Randomly Zoom by 10% some training images

Randomly shift images horizontally by 10% of the width

Randomly shift images vertically by 10% of the height





vertical_flip , horizontal_flip are not appropriate to apply here

since it could have lead to misclassify symetrical numbers such as 6 and 9.

"""

from keras.preprocessing.image import ImageDataGenerator 



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
for x_batch, y_batch in datagen.flow(X_train, Y_train_oh, batch_size=9, shuffle = False):

    print(x_batch.shape)

    print(y_batch.shape)

    break
for x_batch, y_batch in datagen.flow(X_train, Y_train_oh, batch_size=9, shuffle = False):

    # create a grid of 4x4 images

    fig, axes = plt.subplots(3, 3, figsize=(5,5))

    axes = axes.flatten()

    for i in range(0, 9):

        axes[i].imshow(x_batch[i].reshape(28,28), cmap=pyplot.get_cmap('gray'))

        axes[i].set_xticks(())

        axes[i].set_yticks(())

    plt.tight_layout()

    break
# Set a learning rate annealeras the callback function

"""

ref: https://www.twblogs.net/a/5c114f3ebd9eee5e40bb299e/

當評價指標不再提升時，減少學習率。

當學習停滯時，減少2倍或10倍的學習率常常能獲得較好的效果。

該回調函數檢測指標的情況，如果在patience個epoch中看不到模型性能提升，則減少學習率



參數



monitor：被監測的量

factor：每次減少學習率的因子，學習率將以lr=lr*factor的形式減少

patience：當patience個epoch過去而模型性能不提升時，學習率減少的動作會被觸發

mode：‘auto’，‘min’，‘max’之一，在min模式下，如果檢測值不再降低，則觸發學習率減少。

        在max模式下，當檢測值不再上升則觸發學習率減少。

epsilon：閾值，用來確定是否進入檢測值的“平原區“

cooldown：學習率減少後，會經過cooldown個epoch才重新進行正常操作

min_lr：學習率的下限

"""

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
# Build the net

from sklearn.model_selection import train_test_split 



if __name__ == '__main__' and run_model3:

    X_train_s, X_val, Y_train_s, Y_val = train_test_split(X_train, Y_train_oh, test_size=0.13, random_state=42)

    model = Sequential()

    build_lenet5(model, input_shape=X_train_s.shape[1:], dropout=0.15)

    model.summary()

    

    adam = optimizers.Adam()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    

# Fit the model

# About fit_generator vs fit: https://blog.csdn.net/learning_tortosie/article/details/85243310

#X_val

#Y_val

# Use original image to validate

    epochs = 45

    batch_size = 72

    Train_gen_batch = datagen.flow(X_train_s, Y_train_s, batch_size=batch_size)

    #Val_gen_batch = datagen.flow(X_val, Y_val, batch_size=batch_size)

    datagen_no_aug = ImageDataGenerator()

    Val_gen_batch = datagen_no_aug.flow(X_val, Y_val, batch_size=batch_size)

    hist_dict['run_model3'] = model.fit_generator(Train_gen_batch, epochs = epochs, verbose = 2,

                                      steps_per_epoch = X_train.shape[0] // batch_size,

                                      validation_data = Val_gen_batch,

                                      validation_steps = X_val.shape[0] // batch_size,

                                     callbacks=[learning_rate_reduction])



#history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size), epochs = epochs,

#                              validation_data = (X_val,Y_val), verbose = 2,

#                              steps_per_epoch=X_train.shape[0] // batch_size)

                              

                              #, callbacks=[learning_rate_reduction])
if __name__ == '__main__' and run_model3:

    predictions = model_predict(model)

    print(predictions[:5])

    write_preds(predictions, "keras-lenet5-aug.csv")

# Save prediction in file for Kaggle submission

#np.savetxt('mnist-pred.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')





#submission_rf.to_csv('titanic_sk_rf.csv', index=False)
def build_net_advanced(model, input_shape=X_train.shape[1:], dropout=0.25):

    # N' = (N+2P-F)/S + 1

    #S = [1,2,1,2,1]

    #N_input = [28,28,14,10,5]

    #P = [2,0,0,0,0]

    #N = [28,14,10,5,1]

    #F = [i[0] + 2*i[1] - i[3]*(i[2] - 1) for i in zip(N_input, P, N, S)] #[5,2,5,2]

    



    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', strides=1,

                     activation='relu', input_shape=input_shape))

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', strides=2,

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(3,3), strides=1))

    model.add(Dropout(dropout))

    

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', strides=1,

                     activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', strides=2,

                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=1))

    model.add(Dropout(dropout))

              

    model.add(Flatten()) 



    model.add(Dense(256, activation='relu'))

    model.add(Dropout(dropout))

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(dropout))

    #Output: (10)

    model.add(Dense(10, activation='softmax'))
if __name__ == '__main__' and run_model_adv:

    X_train_s, X_val, Y_train_s, Y_val = train_test_split(X_train, Y_train_oh, test_size=0.15, random_state=42)

    model = Sequential()

    build_net_advanced(model, input_shape=X_train_s.shape[1:], dropout=0.3)

    model.summary()

    

    adam = optimizers.Adam()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

    

# Fit the model

# About fit_generator vs fit: https://blog.csdn.net/learning_tortosie/article/details/85243310

# Use original image to validate

#X_val

#Y_val

#history_adv_net

    epochs = 35

    batch_size = 84

    Train_gen_batch = datagen.flow(X_train_s, Y_train_s, batch_size=batch_size)

    #Val_gen_batch = datagen.flow(X_val, Y_val, batch_size=batch_size)

    datagen_no_aug = ImageDataGenerator()

    Val_gen_batch = datagen_no_aug.flow(X_val, Y_val, batch_size=batch_size)

    hist_dict['run_model_adv'] = model.fit_generator(Train_gen_batch, epochs = epochs, verbose = 2,

                                      steps_per_epoch = X_train.shape[0] // batch_size,

                                      validation_data = Val_gen_batch,

                                      validation_steps = X_val.shape[0] // batch_size,

                                     callbacks=[learning_rate_reduction])
if __name__ == '__main__' and run_model_adv:

    predictions = model_predict(model)

    print(predictions[:5])

    write_preds(predictions, "keras-adv-net.csv")

# Plot the loss and accuracy curves for training and validation 

def loss_acc_plt(history):

    fig, ax = plt.subplots(2,1)

    ax[0].plot(history.history['loss'], color='b', label="Training loss")

    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

    legend = ax[0].legend(loc='best', shadow=True)



    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

    legend = ax[1].legend(loc='best', shadow=True)

    

#if __name__ == '__main__' and run_model_adv:

#    loss_acc_plt(history_adv_net)



if run_model1:

    loss_acc_plt(hist_dict['run_model1'])

if run_model2:

    loss_acc_plt(hist_dict['run_model2'])

if run_model3:

    loss_acc_plt(hist_dict['run_model3'])

if run_model_adv:

    loss_acc_plt(hist_dict['run_model_adv'])
from sklearn.metrics import confusion_matrix



_, X_val_check, _, Y_val_check = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)

Ypred_val_check = model_predict_val(model, set_check=X_val_check)



cm = confusion_matrix(Y_val_check.values, Ypred_val_check)

cm
plt.figure(figsize = (10,8))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)



plt.title('Confusion Matrix', fontsize=14)

plt.colorbar()

n_classes = cm.shape[0]

range_class = range(n_classes)

tick_marks = np.arange(len(range_class))

plt.xticks(tick_marks, range_class, rotation=-45, fontsize=14)

plt.yticks(tick_marks, range_class, fontsize=14)

plt.xlabel('Predicted label', fontsize=14)

plt.ylabel('True label', fontsize=14)



for i in range_class:

    for j in range_class:        

        plt.text(j, i, cm[i,j], horizontalalignment="center", fontsize=14, 

                color="white" if i==j else "black")

plt.plot
X_val_check[Ypred_val_check != Y_val_check.values].shape[0] / X_val_check.shape[0]
display_digits(dim = (2,3),

               X = X_val_check[Ypred_val_check != Y_val_check.values],

               Y_true = Y_val_check.values[Ypred_val_check != Y_val_check.values],

               pred = Ypred_val_check[Ypred_val_check != Y_val_check.values])
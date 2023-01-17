import os

import pandas as pd

import numpy as np



from datetime import datetime



import matplotlib.pyplot as plt



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D, BatchNormalization, Activation

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import plot_model



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



import itertools

path_main = '/kaggle/input/Kannada-MNIST'

path_train = os.path.join(path_main, 'train.csv')

path_test = os.path.join(path_main, 'test.csv')

path_sample_sub = os.path.join(path_main, 'sample_submission.csv')

path_dig_mnist = os.path.join(path_main, 'Dig-MNIST.csv')
train_dane = pd.read_csv(path_train)

test_dane = pd.read_csv(path_test)

sample_dane = pd.read_csv(path_sample_sub)

dig_mnist_dane = pd.read_csv(path_dig_mnist)
#

X_train = train_dane.iloc[:,1:].values.astype('float32') # all pixel values

y_train = train_dane.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test_dane.iloc[:,1:].values.astype('float32')

Dig_mnist_im = dig_mnist_dane.iloc[:,1:].values.astype('float32')



X_train = X_train.reshape(X_train.shape[0], 28, 28)   

X_test = X_test.reshape(X_test.shape[0], 28, 28)   

Dig_mnist_im = Dig_mnist_im.reshape(Dig_mnist_im.shape[0], 28, 28)   
plt.figure(figsize=(10, 10))

for idx in range(0, 9):

    plt.subplot(330 + (idx+1))

    plt.imshow(X_train[idx], cmap=plt.get_cmap('gray'))

    plt.title(y_train[idx])

    

    plt.tight_layout
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)    

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)



mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px



y_train= to_categorical(y_train)

num_classes = y_train.shape

X = X_train

y = y_train

X_traine, X_val, y_traine, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
input_shape=(28,28,1)
model = Sequential([

    Conv2D(32, (3,3), input_shape=input_shape, padding='same', activation='relu'),

    Conv2D(32, (3,3), padding='same', activation='relu'),

    BatchNormalization(momentum=0.15),

    MaxPool2D((2,2)),

    BatchNormalization(momentum=0.15),

    Conv2D(64, (5,5), padding='same', activation='relu'),

    Dropout(0.3),

    

    Conv2D(32, (3,3), padding='same', activation='relu'),

    Conv2D(32, (3,3), padding='same', activation='relu'),

    BatchNormalization(momentum=0.15),

    MaxPool2D((2,2)),

    BatchNormalization(momentum=0.15),

    Conv2D(64, (5,5), padding='same', activation='relu'),

    Dropout(0.3),



    

    Flatten(),

    

    Dense(128, activation='relu'),

    Dropout(0.4),

    Dense(10, activation='softmax')

])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
earlystopper = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=0, mode='max', baseline=0.995, restore_best_weights=False)



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001) #0.00001

callback = [learning_rate_reduction]
# I discovered when playing around with MNIST dataset that batch size = 256 is sufficient is the best for it and I think here it is also true.

history = model.fit(

    X_traine, y_traine,

    batch_size=512,

    epochs=50,

    validation_data=(X_val, y_val),

    callbacks=callback

)
test_im = X_train[6]

plt.imshow(test_im.reshape(28,28), cmap='viridis', interpolation='none')

from keras.models import Model

layer_outputs = [layer.output for layer in model.layers[:len(model.layers)]] #I added here len(model.layers) so I will always get proper numbeer of layers which will be checked later. 

activation_model = Model(input=model.input, output=layer_outputs)

activations = activation_model.predict(test_im.reshape(1,28,28,1))



first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
print(model.input)
model.layers[:-1]# Droping The Last Dense Layer
layer_names = []

for layer in model.layers[:-1]:

    layer_names.append(layer.name) 

images_per_row = 16

zipped_layers = zip(layer_names, activations)

for layer_name, layer_activation in zipped_layers: #this loop     

    if layer_name.startswith('conv'):

        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):

            for row in range(images_per_row):

                channel_image = layer_activation[0,:, :, col * images_per_row + row]

                channel_image -= channel_image.mean()

                channel_image /= channel_image.std()

                channel_image *= 64

                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size,

                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],

                            scale * display_grid.shape[0]))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')
layer_names = []

for layer in model.layers[:-1]:

    layer_names.append(layer.name) 

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):

    if layer_name.startswith('max'):

        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // images_per_row

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):

            for row in range(images_per_row):

                channel_image = layer_activation[0,:, :, col * images_per_row + row]

                channel_image -= channel_image.mean()

                channel_image /= channel_image.std()

                channel_image *= 64

                channel_image += 128

                channel_image = np.clip(channel_image, 0, 255).astype('uint8')

                display_grid[col * size : (col + 1) * size,

                             row * size : (row + 1) * size] = channel_image

        scale = 1. / size

        plt.figure(figsize=(scale * display_grid.shape[1],

                            scale * display_grid.shape[0]))

        plt.title(layer_name)

        plt.grid(False)

        plt.imshow(display_grid, aspect='auto', cmap='viridis')
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
# Look at confusion matrix 

#Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred, axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val, axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10))
print(history.history.keys())

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes - Y_true != 0)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = X_val[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))



# Difference between the probability of the predicted label and the true label

delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors



# Sorted list of the delta prob errors

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
predictions = model.predict_classes(X_test, verbose=0)





predictions = model.predict_classes(X_test, verbose=0)



submission = pd.read_csv(path_sample_sub)

submission['label'] = predictions



submission.to_csv('submission.csv', index=False)
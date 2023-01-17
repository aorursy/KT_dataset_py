import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib.pyplot as plt # plots

from keras.utils.np_utils import to_categorical # One-hot encoding

from keras.preprocessing.image import ImageDataGenerator # Data augmentation

from sklearn.model_selection import train_test_split # Split training data into training and validation sets

# Load the dataset

train = pd.read_csv("../input/train.csv")
# Preview the data structure

print(train.shape)

print(train.info())

train.head()
# Load the test data

test = pd.read_csv("../input/test.csv")

print(test.info())

test.head()
# Separate the predictors (aka pixel values) from the targets

# Convert the type to float so we can preserve the precision when normalizing the values

X_train = train.drop(['label'],axis=1).astype('float32')

y_train = train['label'].astype('float32')

X_test = test.values.astype('float32')
images = X_train.values.reshape(X_train.shape[0], 28, 28)

print(images.shape)



def display_image_data(img, index):

    plt.subplot(3, 3, index + 1)

    plt.imshow(img, cmap=plt.get_cmap('gray'))

    plt.title(y_train[index].astype('int32'))



plt.figure(figsize=(15,10))



for index, image in enumerate(images[0:9]):

    display_image_data(image, index)
def render_pixel_values(image, plot):

    plot.imshow(image, cmap='gray')

    width, height = image.shape

    threshold = image.max() / 2

    

    # Loop through all the pixel values

    for x in range(width):

        for y in range(height):

            pixel_value = str(round(image[x][y], 2).astype('int32'))

            

            # If background is rendered black, render the text white and vice versa

            text_color = 'white' if image[x][y] < threshold else 'black'

            plot.annotate(pixel_value,

                          xy=(y,x),

                          horizontalalignment='center',

                          verticalalignment='center',

                          color=text_color)



fig = plt.figure(figsize = (12,12)) 

subplot = fig.add_subplot(1, 1, 1)



# Show the first image with the details

render_pixel_values(images[0], subplot)
# Add the dimension for the color channel (grey)

X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

print(X_train.shape)



X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

print(X_test.shape)
# Normalize the values

def normalize(m):

    return m / 255



X_train = normalize(X_train)

X_test = normalize(X_test)
# One-hot encode the labels

print('Labels')

print(y_train[:5])

y_train = to_categorical(y_train, 10)

print('Encoded labels')

print(y_train[:5])
# Build the model 

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout # Modeling

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.layers.normalization import BatchNormalization



# Save the weights which produce the best validation accuracy

checkpoint = ModelCheckpoint(filepath='mnist.model.best.hdf5', 

                             verbose=1,

                             save_best_only=True,

                             monitor='val_acc')

def build_model():

    model = Sequential([

        Convolution2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),

        BatchNormalization(),

        

        Convolution2D(16, (3,3), activation='relu'),

        BatchNormalization(),

        MaxPooling2D(),

        Dropout(0.25),

        

        Convolution2D(32, (3,3), activation='relu'),

        # BatchNormalization(),

        Convolution2D(32, (3,3), activation='relu'),

        BatchNormalization(),

        MaxPooling2D(),

        Dropout(0.25),

        

        Flatten(),

        Dense(256, activation='relu'),

        BatchNormalization(),

        Dropout(0.25),

        Dense(10, activation='softmax')

    ])

    

    model.compile(optimizer=Adam(),

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model



model = build_model()

model.summary()
from keras.utils import plot_model

from IPython.display import Image



plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

Image("model.png")
X_train, X_val, y_train, y_val = train_test_split(X_train,

                                                  y_train,

                                                  test_size=0.1,

                                                  random_state=42)
# Reduce the learning rate when the current value cannot minimize the cost function anymore...

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)


# Create augemented data with Keras ImageDataGenerator

image_generator = ImageDataGenerator(featurewise_center=False, # set input mean to 0 over the dataset

                                     samplewise_center=False,  # set each sample mean to 0

                                     featurewise_std_normalization=False,  # divide inputs by std of the dataset

                                     samplewise_std_normalization=False,  # divide each input by its std

                                     zca_whitening=False,  # apply ZCA whitening

                                     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

                                     zoom_range = 0.1, # Randomly zoom image 

                                     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

                                     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

                                     horizontal_flip=False,  # randomly flip images

                                     vertical_flip=False)  # randomly flip images)



# 

batch_size = 96



epochs = 60



# steps_per_epoch = the number of batch iterations before a training epoch is considered finished

# We want to use every sample so divide training set with batch size

# Total samples = epochs * steps * batchsize

steps_per_epoch = X_train.shape[0] / batch_size



batches = image_generator.flow(X_train, y_train, batch_size=batch_size)
# Train the actual network

history = model.fit_generator(generator=batches,

                              steps_per_epoch=steps_per_epoch,

                              epochs=epochs,

                              validation_data=(X_val, y_val),

                              callbacks=[checkpoint, learning_rate_reduction])
# Load the best weights

model.load_weights('mnist.model.best.hdf5')
plt.plot(history.history['acc'], label='train')

plt.plot(history.history['val_acc'], label='test')

plt.legend()

plt.show()
_, train_acc = model.evaluate(X_train, y_train, verbose=0)

_, test_acc = model.evaluate(X_val, y_val, verbose=0)

print('Train accuracy: %.3f, Test accuracy: %.3f' % (train_acc, test_acc))
import seaborn as sns

from sklearn.metrics import confusion_matrix



def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    cm_dataframe = pd.DataFrame(confusion_matrix,

                         index=class_names,

                         columns=class_names)

    

    fig = plt.figure(figsize=figsize)

    

    heatmap = sns.heatmap(cm_dataframe,

                          annot=True,

                          fmt="d",

                          cbar=False,

                          vmin=0.0,

                          vmax=3.0)

    

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)

    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



Y_pred = model.predict(X_val)



# Take the index of the highest probability ( = predicted class)

Y_pred_classes = np.argmax(Y_pred, axis = 1)



# The true labels of the images

y_true = np.argmax(y_val, axis = 1)



confusion_mtx = confusion_matrix(y_true, Y_pred_classes) 

print_confusion_matrix(confusion_mtx, range(10))
errors = (Y_pred_classes - y_true != 0)



# The predicted class which were incorrect

Y_pred_classes_errors = Y_pred_classes[errors]



# The incorrect predictions (array of probability vectors)

Y_pred_errors = Y_pred[errors]



# The correct labels of the incorrect predictions

Y_true_errors = y_true[errors]

# Probabilities of the wrong predicted numbers

Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)



# Predicted probabilities of the true values in the error set

true_prob_errors = [Y_pred_errors[i][y_i] for i, y_i in enumerate(Y_true_errors)]



# Get the three most incorrect predictions

most_incorrect_preds = np.argsort(Y_pred_errors_prob - true_prob_errors)[-3:]



def display_error_predictions(errors_index,img_errors,pred_errors, obs_errors):

    for col in range(3):

        error = errors_index[col]

        plt.subplot(1, 3, col + 1)

        plt.imshow((img_errors[error]).reshape((28,28)), cmap=plt.get_cmap('gray'))

        plt.title("Prediction:{}\nTrue label:{}".format(pred_errors[error],obs_errors[error]))



display_error_predictions(most_incorrect_preds, X_val[errors], Y_pred_classes_errors, Y_true_errors)
predictions = model.predict_classes(X_test, verbose=2)
sub = pd.read_csv('../input/sample_submission.csv')

sub['Label'] = predictions

sub.to_csv('submission.csv',index=False)
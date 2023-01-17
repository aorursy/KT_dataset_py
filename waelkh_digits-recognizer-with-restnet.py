!pip install --upgrade "tensorflow==1.14" "keras>=2.0"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist

import time
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers import Add, AveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

sns.set(style='white', context='notebook', palette='deep')
def loss_acc_plot(model_eval):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(model_eval.history['loss'], color='b', label="Training loss")
    ax[0].plot(model_eval.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(model_eval.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(model_eval.history['val_accuracy'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

# Draw a confusion matrix that can be used to observe high false positives

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

def plotImages(images_arr,labels):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img,label, ax in zip( images_arr,labels, axes):
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)
    plt.tight_layout()
    plt.show()

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
def residual_block(inputs, filters, strides=1):
    """Residual block
    
    Shortcut after Conv2D -> ReLU -> BatchNorm -> Conv2D
    
    Arguments:
        inputs (tensor): input
        filters (int): Conv2D number of filterns
        strides (int): Conv2D square stride dimensions

    Returns:
        x (tensor): input Tensor for the next layer
    """
    y = inputs # Shortcut path
    
    # Main path
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='same',
    )(x)
    x = BatchNormalization()(x)
    
    # Fit shortcut path dimenstions
    if strides > 1:
        y = Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding='same',
        )(y)
        y = BatchNormalization()(y)
    
    # Concatenate paths
    x = Add()([x, y])
    x = Activation('relu')(x)
    
    return x
    
    
def resnet(input_shape, num_classes, filters, stages):
    """ResNet 
    
    At the beginning of each stage downsample feature map size 
    by a convolutional layer with strides=2, and double the number of filters.
    The kernel size is the same for each residual block.
    
    Arguments:
        input_shape (3D tuple): shape of input Tensor
        filters (int): Conv2D number of filterns
        stages (1D list): list of number of resiual block in each stage eg. [2, 5, 5, 2]
    
    Returns:
        model (Model): Keras model
    """
    # Start model definition
    inputs = Input(shape=input_shape)
    x = Conv2D(
        filters=filters,
        kernel_size=7,
        strides=1,
        padding='same',
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Stack residual blocks
    for stage in stages:
        x = residual_block(x, filters, strides=2)
        for i in range(stage-1):
            x = residual_block(x, filters)
        filters *= 2
        
    # Pool -> Flatten -> Classify
    x = AveragePooling2D(4)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(int(filters/4), activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model 

def train_model(epochs, filters, stages, batch_size):
    """Helper function for tuning and training the model
    
    Arguments:
        epoch (int): number of epochs
        filters (int): Conv2D number of filterns
        stages (1D list): list of number of resiual block in each stage eg. [2, 5, 5, 2]
        batch_size (int): size of one batch
        visualize (bool): if True then plot training results 
    
    Returns:
        model (Model): Keras model
    """
    # Create and compile model
    model = resnet(
        input_shape=X_train[0].shape,
        num_classes=Y_train[0].shape[-1],
        filters=filters, 
        stages=stages
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
        metrics=['accuracy']
    )

    # Define data generator
    datagen = ImageDataGenerator(  
        rotation_range=10,  
        zoom_range=0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1
    )
    datagen.fit(X_train)

    # Fit model
    history = model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        validation_data=(X_val, Y_val),
        epochs=epochs, 
        verbose=1, 
        workers=12
    )

    return model
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
print ('train:',train.shape,'test:',test.shape)

Y_train = train["label"]
X_train = train.drop(labels = ["label"], axis = 1) 

# train with additional public datasets fr the same task
(x_train_external, y_train_external), (x_test_external, y_test_external) = mnist.load_data()

train_external = np.concatenate([x_train_external, x_test_external], axis=0)
y_train_external = np.concatenate([y_train_external, y_test_external], axis=0)
Y_train_external = y_train_external
X_train_external = train_external.reshape(-1, 28*28)
# The distibution of classes
g = sns.countplot(Y_train)
# Normalize data to make CNN faster
X_train = X_train / 255.0
test = test / 255.0
X_train_external = X_train_external / 255.0
# Reshape Picture is 3D array (height = 28px, width = 28px , canal = 1)
X_train = np.concatenate((X_train.values, X_train_external))
Y_train = np.concatenate((Y_train, Y_train_external))

X_train = X_train.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
# Split dataset into training set and validation set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)

# Draw an example of a data set to see
print(X_train[:4].shape, X_train[:4][:,:,:,0].shape)
plotImages(X_train[:5][:,:,:,0],Y_train[:5])
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(10, activation = "softmax"))

# Define Optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
#model.summary()

# print out model look
#from keras.utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#from IPython.display import Image
#Image("model.png")
from tensorflow.keras.callbacks import EarlyStopping
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
#Adjusting epochs and batch_size
epochs = 55
batch_size = 128
#Data Augmentation 
datagen = ImageDataGenerator(
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

#val_datagen = ImageDataGenerator(
#        featurewise_center=True)  # randomly flip images
#val_datagen.fit(X_val)
#val_datagen.fit(test)
#Prediction model
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001, cooldown=0)
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              ,callbacks=[lr_schedule])

# Draw the loss and accuracy curves of the training set and the validation set.
# Can judge whether it is under-fitting or over-fitting
loss_acc_plot(history)
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Show some wrong results, and the difference between the predicted label and the real labe
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

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
# Make predictions about test sets
results = model.predict(test)

# Convert one-hot vector to number
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
# Save the final result in cnn_mnist_submission.csv
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_restnet_submission.csv",index=False)

#simple_model = resnet(
#    input_shape=X_train[0].shape, 
#    num_classes=Y_train[0].shape[-1], 
#    filters=64, 
#    stages=[2]
#)
#simple_architecture = plot_model(simple_model, show_shapes=True, show_layer_names=False)
#simple_architecture.width = 600
#simple_architecture
# Train models
#models = []
#for i in range(1):
#    print('-------------------------')
#    print('Model: ', i+1)
#    print('-------------------------')
#    model = train_model(
#        epochs=10,
#        filters=64,
#        stages=[5, 3, 3],
#        batch_size=128
#    )
#    models.append(model)
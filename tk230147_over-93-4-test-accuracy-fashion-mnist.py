#import mnist_reader

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

# or notebook

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model



sns.set(style='white', context='notebook', palette='deep')
# Load the data

train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")



training =np.array(train,dtype ='float32') # only used for display in grid
#view of images in grid format

# Define the dimensions of the plot grid 



w_grid=15

l_grid=15



# fig,axes = plt.subplot(l_grid,w_grid)

# subplot return the figure object and axes object

# we can use the axes object to plot specific figures at various locations



fig,axes=plt.subplots(l_grid,w_grid,figsize=(17,17))



axes = axes.ravel() # flatten thr 15 X 15 matrix into 225 array 



n_training = len(training) # get the length of the training dataset



#select a random number from 0 t n_training



for i in np.arange(0,w_grid*l_grid): #create evenly spaces variables

    #select a random number

    

    index = np.random.randint(0,n_training)

    # read and disply and images with the selectd index

    axes[i].imshow(training[index,1:].reshape((28,28)))

    axes[i].set_title(training[index,0],fontsize = 8)

    axes[i].axis('off')

    

plt.subplots_adjust(hspace=0.4)
Y_train = train["label"]

Y_test = test["label"]



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 

X_test = test.drop(labels = ["label"],axis = 1) 



# free some space

#del train 

#del test 



g = sns.countplot(Y_train)



Y_train.value_counts()
# Check the data

X_train.isnull().any().describe()
X_test.isnull().any().describe()
# Normalize the data

X_train = X_train / 255.0

X_test = X_test / 255.0
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



print(X_train.shape)

print(X_test.shape)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)

Y_test = to_categorical(Y_test, num_classes = 10)
# Set the random seed

random_seed = 2
# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
# Some examples

g = plt.imshow(X_train[11][:,:,0])
# Set the CNN model 

# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.15))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.15))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.3))

model.add(Dense(10, activation = "softmax"))
# Compile the model

model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 15

batch_size = 32
# Without data augmentation i obtained an accuracy of 0.98114

#history = model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, 

#          validation_data = (X_val, Y_val), verbose = 2)
# With data augmentation to prevent overfitting (accuracy 0.99286)



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



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

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 



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
# Predict the values from the test dataset

Y_pred_test = model.predict(X_test)

# Convert predictions classes to one hot vectors 

Y_pred_classes_test = np.argmax(Y_pred_test,axis = 1) 

# Convert test observations to one hot vectors

Y_true_test = np.argmax(Y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true_test, Y_pred_classes_test) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
results=model.evaluate(X_test, Y_test)
print('test loss, test acc:', results)
model.summary()
# save model and architecture to single file

#model.save("model.h5")

#print("Saved model to disk")
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
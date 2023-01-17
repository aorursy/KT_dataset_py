'''Importing Data Manipulattion Moduls'''

import numpy as np

import pandas as pd



'''Seaborn and Matplotlib Visualization'''

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



'''Importing preprocessing libraries'''

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown

def bold(string):

    display(Markdown(string))
'''Importing tensorflow libraries'''

import tensorflow as tf 

print(tf.__version__)



from tensorflow.keras import layers, models
'''Read in train and test data from csv files'''

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

sample_sub = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
'''Train and test data at a glance.'''

bold('**Preview of Train Data:**')

display(train.head(3))

bold('**Preview of Test Data:**')

display(test.head(3))
'''Ckecking for null and missing values'''

bold('**Train Data**')

display(train.isnull().any(). describe())

bold('**Test Data**')

display(test.isnull().any(). describe())
'''Seting X and Y'''

y_train = train['label']



# Drop 'label' column

X_train = train.drop('label', axis = 1)



X_test = test.drop('id', axis = 1)
"""Let's have a final look at our data"""

bold('**Data Dimension for Model Building:**')

print('Input matrix dimension:', X_train.shape)

print('Output vector dimension:',y_train.shape)

print('Test data dimension:', X_test.shape)
'''Visualizating the taget distribution'''

plt.figure(figsize = (8,8))

sns.countplot(y_train, palette='Paired')

plt.show()
images = train.iloc[:,1:].values

images = images.astype(np.float)



# convert from [0:255] => [0.0:1.0]

images = np.multiply(images, 1.0 / 255.0)



image_size = images.shape[1]

print('image_size => {0}'.format(image_size))



# in this case all images are square

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)



print('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))
'''Displaying image'''

# display image

def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(image_width,image_height)

    

    plt.axis('off')

    plt.imshow(one_image, cmap='binary')



# output image     

display(images[8])
'''Normalizing the data'''

X_train = X_train / 255.0

X_test = X_test / 255.0
'''convert class labels from scalars to one-hot vectors'''

# 0 => [1 0 0 0 0 0 0 0 0 0]

# 1 => [0 1 0 0 0 0 0 0 0 0]

# ...

# 9 => [0 0 0 0 0 0 0 0 0 1]

y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10, dtype='uint8')
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)
'''Set the random seed'''

seed = 44

'''Split the train and the validation set for the fitting'''

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=seed)
'''Set the CNN model'''

# CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = models.Sequential()

model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28,28,1)))

model.add(layers.Conv2D(32, (5, 5), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))

          

model.add(layers.Conv2D(64, (5, 5), activation='relu'))

model.add(layers.Conv2D(64, (5, 5), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.25))

          

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.Dense(10, activation='softmax'))
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

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
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(X_train, y_train, batch_size = 1000, epochs = 10, validation_data = (X_val, y_val), verbose = 2)
'''Training and validation curves'''

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
'''confusion matrix'''

import seaborn as sns

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()
'''predict results'''

results = model.predict(X_test)



'''select the indix with the maximum probability'''

results = np.argmax(results,axis = 1)
sample_sub['label'] = results

sample_sub.to_csv('submission.csv',index=False)
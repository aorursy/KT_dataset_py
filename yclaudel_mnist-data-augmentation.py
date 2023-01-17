import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split



import matplotlib

import matplotlib.pyplot as plt
# Utils

def print_metrics(y_train,y_pred):

    conf_mx = confusion_matrix(y_train,y_pred)

    print(conf_mx)

    print ("------------------------------------------")

    print (" Accuracy    : ", accuracy_score(y_train,y_pred))

    print ("------------------------------------------")



def shift_image(X, dx, dy,length=28):

    X=X.reshape(length,length)

    X = np.roll(X, dy, axis=0)#horizontal

    X = np.roll(X, dx, axis=1)#vertical

    return X.reshape([-1])



def print_image(flat_image,length=28):

    plt.imshow(flat_image.reshape(length, length), cmap = matplotlib.cm.binary,interpolation="nearest")

    plt.axis("off")

    plt.show()  

# Load the data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv").values

y = train["label"].values

X = train.drop(labels = ["label"],axis = 1).values

print("Value Counts :")

print(train["label"].value_counts())

del train 



# we want value between [0,1]

X = X / 255.0

test = test / 255.0

# print Dimension

print("dim(X)    = ",X.shape)

print("dim(y)    = ",y.shape)

print("dim(test) = ",test.shape)
DATA_AUGMENTED_WITH_SHIFT = False
if DATA_AUGMENTED_WITH_SHIFT:

    X_augmented = [image for image in X]

    y_augmented = [label for label in y]



    for dx, dy in ((1,1),(-1,-1),(-1,1),(1,-1)):

        for image, label in zip(X, y):

            X_augmented.append(shift_image(image, dx, dy))

            y_augmented.append(label)



        

    X_augmented = np.array(X_augmented)

    y_augmented = np.array(y_augmented)

    print("   X_augmented Dimension : ",X_augmented.shape)

    shuffle_idx = np.random.permutation(len(X_augmented))

    X_augmented = X_augmented[shuffle_idx]

    y_augmented = y_augmented[shuffle_idx]



    X_train = X_augmented.reshape(-1,28,28,1)

    test = test.reshape(-1,28,28,1)

    Y_train = to_categorical(y_augmented, num_classes = 10)

else:

    X_train = X.reshape(-1,28,28,1)

    test = test.reshape(-1,28,28,1)

    Y_train = to_categorical(y, num_classes = 10)



print("dim(X_train)    = ",X_train.shape)

print("dim(Y_train)    = ",Y_train.shape)

print("dim(test)       = ",test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)

# define data preparation

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



# fit parameters from data

datagen.fit(X_train)
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

epochs = 30

batch_size = 71



model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])

# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

#confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

print_metrics(Y_true, Y_pred_classes)
# predict results

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
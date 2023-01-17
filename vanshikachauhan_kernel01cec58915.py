#declare headers

import pandas as pd

import numpy as np

from keras import models

# convert to one-hot-encoding

from keras.utils.np_utils import to_categorical 

from keras import layers

from keras.preprocessing import image

import matplotlib.pyplot as plt

import matplotlib.image as mpimg



# Load the Test data

train = pd.read_csv("../input/train.csv")

print("Train dataset shape",train.shape)

Y_train = train["label"]

# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)
# Load the Test data

test = pd.read_csv("../input/test.csv")

print("Test dataset shape",test.shape)

test.head()
# Normalize the data

X_train = X_train / 255.0

test = test / 255.0



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("Train image shape after reshaping:",X_train.shape)

print("Test image shape after reshaping:",test.shape)
# Encode labels to one hot vectors (ex : 3 -> [0,0,1,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 10)
# Some examples

g = plt.imshow(X_train[7][:,:,0])

# Define model architecture

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))



model.add(layers.Flatten())

#6. Add a fully connected layer and then the output layer model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

#Summary of model

model.summary()
#8. Compile model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',

metrics=["categorical_accuracy"])
#9. Fit model on training data

model.fit(X_train, Y_train,  batch_size=64, epochs=5, verbose=1)
#Predicting the classes and display 

for i in range(1,10) :

    img1=test[i][:,:,0]

    img1=image.img_to_array(img1)

    img1.shape

    tests1=img1.reshape((1,28,28,1))#in conv networks the should pe reshaped into 4d tensors

    img_class=model.predict_classes(tests1)

    prediction=img_class[0]

    classname=img_class[0]

    print(" Predicted classname:",classname)



    #displaying the digit and classifying digit

    img1=img1.reshape((28,28))

    plt.imshow(img1)

    plt.title(classname)

    plt.show()
# predict results

res = model.predict(test)



# select the indix with the maximum probability

res = np.argmax(res,axis = 1)



res = pd.Series(res,name="Label")
#Submitting the predictions

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),res],axis = 1)



submission.to_csv("predict_digit_class.csv",index=False)
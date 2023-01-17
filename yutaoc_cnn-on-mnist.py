import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import keras



from sklearn.model_selection import train_test_split

# Load the date set

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

print('The dimension of training data is: {}'.format(train.shape))

print('The dimension of testing data is: {}'.format(test.shape))

# Separate the label and image

train_label = train['label']

train_img = train.drop(columns='label')

test_img = test

del train

del test

axes = sns.countplot(train_label)  # visulize the training labels

# Normalization

train_img = train_img / 255.0

test_img = test_img / 255.0

# Reshape the data

train_img = train_img.values.reshape(-1,28,28,1)

test_img = test_img.values.reshape(-1,28,28,1)

print('The dimension of the training data after reshape is: {}'.format(train_img.shape))

print('The dimension of the testing data after reshape is: {}'.format(test_img.shape))

# Covert label to hot map

train_label = keras.utils.np_utils.to_categorical(train_label, num_classes=10)

print('One example of the hot map is:\nThe label --> {}\nconverting to --> {}'.

      format(np.nonzero(train_label[0])[0][0], train_label[0]))
# Create the train set and validation set

train_img, vali_img, train_label, vali_label = train_test_split(train_img, train_label, 

                                                                test_size = 0.15, random_state=2)

print('The size of the train set is: {}'.format(train_img.shape[0]))

print('The size of the validate set is: {}'.format(vali_img.shape[0]))
# Visualize one sample of the data

# Sample images from the train set

for i in range(5):

    plt.subplot(2, 5, i+1)

    plt.imshow(train_img[i][:,:,0], cmap='gray')

    plt.title("label :{}".format(np.nonzero(train_label[i])[0][0]))

# Sample images from the test set

for i in range(5):

    plt.subplot(2, 5, 5+i+1)

    plt.imshow(vali_img[i][:,:,0], cmap='gray')

    plt.title("label :{}".format(np.nonzero(vali_label[i])[0][0]))

plt.subplots_adjust(left = 0.125, right = 2, top = 1.5, bottom = 0.1)

# Define the model

model = keras.models.Sequential()  # create the model



model.add(keras.layers.Conv2D(filters = 6, kernel_size = (5,5),padding = 'Same', 

                             activation ='relu', input_shape = (28,28,1)))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))



model.add(keras.layers.Conv2D(filters = 16, kernel_size = (5,5),padding = 'Same', 

                             activation ='relu',))

model.add(keras.layers.Dropout(rate=0.15))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))



model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))



model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(keras.layers.Dropout(rate=0.15))

model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))



model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256, activation=tf.nn.relu))

model.add(keras.layers.Dropout(rate=0.5))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))



model.summary()

# Other parameters setting for the model

opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=opt,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

history = model.fit(train_img, train_label, batch_size = 32, epochs = 25, 

                    validation_data = (vali_img, vali_label), verbose = 2)

# Plot the learning curve

plt.subplot(2, 1, 1)

plt.plot(history.history['loss'], color='b', label="Training loss")

plt.plot(history.history['val_loss'], color='r', label="validation loss")

plt.legend(loc='best', shadow=True)



plt.subplot(2, 1, 2)

plt.plot(history.history['acc'], color='b', label="Training accuracy")

plt.plot(history.history['val_acc'], color='r',label="Validation accuracy")

plt.legend(loc='best', shadow=True)

# Dsiplay some incorrectly classified images

pred_prob = model.predict(vali_img)  # prob. of different classes

pred_class = np.argmax(pred_prob,axis = 1)  # predicted classes

true_class = np.argmax(vali_label,axis = 1)  # actual classes



errors = (pred_class - true_class != 0)  # wrong prediction marked as 1



# Take out only the wrong predictions

pred_prob_wrong = pred_prob[errors] 

pred_class_wrong = pred_class[errors]

true_class_wrong = true_class[errors]

vali_img_wrong = vali_img[errors]



pred_prob_wrong_top = np.max(pred_prob_wrong, axis=1)  # top prob.

true_prob_wrong = np.diagonal(np.take(pred_prob_wrong, true_class_wrong, axis=1))  # prob. of its true label

delta = pred_prob_wrong_top - true_prob_wrong  # differences

delta_sort = np.argsort(delta)  # sort (increase)

top10 = delta_sort[-10:]  # take the top ten (maximum difference)



# plot the top ten

for i in range(10):

    plt.subplot(2, 5, i+1,)

    error = top10[i]

    plt.imshow(vali_img_wrong[error].reshape(28, 28), cmap='gray')

    plt.title("Predicted label :{}\nTrue label :{}".format(pred_class_wrong[error],true_class_wrong[error]))

plt.subplots_adjust(left = 0.125, right = 2, top = 1.5, bottom = 0.1)

# Predict on testing data

pred = model.predict(test_img)

pred_label = np.argmax(pred,axis = 1) # choose the label with max prob. as predicted label

# Generate the submission file

result = pd.Series(pred_label, name = "Label")

img_id = pd.Series(range(1,28001),name = "ImageId")

submission = pd.concat([img_id, result],axis = 1)

submission.to_csv("result.csv",index = False)

# show the first five line of the file

submission.head()

# Importing all necessary libraries

import pandas as pd

import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import os



from keras.models import Sequential

from keras.layers import Dense

from keras.utils.np_utils import to_categorical



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



warnings.filterwarnings('ignore')
# Importing datasets

train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')



# I want to use Dig-MNIST for final validation

val_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

sub_df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')



# Pandas conversts data to int64 dtype, so, to reduce memory usage, I'll convert it to uint8 dtype

train_df = train_df.astype(np.uint8)

test_df = test_df.astype(np.uint8)

val_df = val_df.astype(np.uint8)
print(train_df.info(), '\n')

print(test_df.info(), '\n')

print(val_df.info())
train_df.head()
# test_df has 'id' column, we need to drop it before predicting

test_df.head()
val_df.head()
fig = plt.figure(figsize = (13, 4))

for i, dataset, name in ((1, train_df, 'train_df'), (2, val_df, 'val_df')):

    plt.subplot(f'12{i}')

    counts = dataset['label'].value_counts()

    sns.barplot(x = counts.index, y = counts.values).set_title(name)
# Creating function for plotting

def plot_images(dataset, figsize = (17, 17)):

    

    '''

    Plots 100 images from selected dataset in 10x10 shape.

    '''

    

    fig = plt.figure(figsize = figsize)

    for i in range(10):

        data = dataset[dataset['label'] == i].drop('label', axis = 1) # Data, that contains only i'th labels



        for j in range(10):

            ax = fig.add_subplot(10, 10, int(f'{i}{j}') + 1)

            img = data.sample(1) # I'm taking random sample from data to plot

            index = img.index[0] # Index for title

            img = np.array(img).reshape((28, 28)) # To plot our image, we need to reshape it to 28x28

            plt.imshow(img, cmap = 'gray') # Plot image

            plt.axis('off') # Don't show X and Y axes 

            ax.set_title(f'{i} ({index})') # Set plot title



    plt.tight_layout() # Doesn't allow our plots overlap each other
# Train_df images

plot_images(train_df)
# Dig-MNIST images

plot_images(val_df)
# Data preparation

# Train data

Y = train_df['label']

X = train_df.drop('label', axis = 1)



# Validation data

y_val = val_df['label']

X_val = val_df.drop('label', axis = 1)



# Dropping 'id' column in test dataset

test_df = test_df.drop('id', axis = 1)



# Normalize the data

X = X / 255.0

X_val = X_val / 255.0

test_df = test_df / 255.0



# One-hot encoding of train features

Y = to_categorical(Y, num_classes = 10)
# Creating train and test datasets for model training

# I'm using stratify to ensure that we have equal proportion of samples from each class in our datasets

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 666) 

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Dictionaries to store our models results

results = {} # Accuracy and loss

preds = {} # Model predictions from val_df

epochs = 20
# Base model

name = 'Base_Model'



# Crating and training model

model = Sequential()

model.add(Dense(10, input_shape = (784, ), activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = epochs, batch_size = 64, validation_data = [X_test, y_test], verbose = 0)



# Appending results to dictionaries

results[name] = history.history

preds[name] = model.predict(X_val)



# Creating predictions for test_df and submission file

sub_preds = model.predict_classes(test_df)

id_col = np.arange(sub_preds.shape[0])

submission = pd.DataFrame({'id': id_col, 'label': sub_preds})

submission.to_csv(f'{name}.csv', index = False)  



print('Done')
# 1 hidden layer model



nodes = (128, 256, 397, 512, 1024)

activations = ('relu', 'sigmoid')



for node in nodes:

    for act in activations:

        name = f'1_hidden_{node}_nodes_{act}_activation'

        print(f'Training: {name}')

        

        model = Sequential()       

        if act == 'relu':            

            model.add(Dense(node, input_shape = (784, ), activation = 'relu', kernel_initializer='he_normal'))

        else:

            model.add(Dense(node, input_shape = (784, ), activation = 'sigmoid', kernel_initializer='glorot_uniform'))        

        model.add(Dense(10, activation = 'softmax'))        

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        

        history = model.fit(X_train, y_train, epochs = epochs, batch_size = 64, validation_data = [X_test, y_test], verbose = 0)

        

        # Appending results to dictionaries

        results[name] = history.history

        preds[name] = model.predict(X_val)

        

        # Creating predictions for test_df and submission file

        

        sub_preds = model.predict_classes(test_df)

        id_col = np.arange(sub_preds.shape[0])

        submission = pd.DataFrame({'id': id_col, 'label': sub_preds})

        submission.to_csv(f'{name}.csv', index = False)       
# 2 hidden layers

nodes = (128, 256, 397, 512, 1024)

activations = ('relu', 'sigmoid')



for node in nodes:

    for act in activations:

        name = f'2_hidden_{node}_nodes_{act}_activation'

        print(f'Training: {name}')

        

        model = Sequential()        

        if act == 'relu':

            model.add(Dense(node, input_shape = (784, ), activation = 'relu', kernel_initializer='he_normal'))

            model.add(Dense(node, activation = 'relu', kernel_initializer='he_normal'))

        else:

            model.add(Dense(node, input_shape = (784, ), activation = 'sigmoid', kernel_initializer='glorot_uniform'))

            model.add(Dense(node, activation = 'sigmoid', kernel_initializer='glorot_uniform'))        

        model.add(Dense(10, activation = 'softmax'))        

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        

        history = model.fit(X_train, y_train, epochs = epochs, batch_size = 64, validation_data = [X_test, y_test], verbose = 0)

        

        # Appending results to dictionaries

        results[name] = history.history

        preds[name] = model.predict(X_val)

        

        # Creating predictions for test_df and submission file

        sub_preds = model.predict_classes(test_df)

        id_col = np.arange(sub_preds.shape[0])

        submission = pd.DataFrame({'id': id_col, 'label': sub_preds})

        submission.to_csv(f'{name}.csv', index = False)  
for key in results.keys():

    fig = plt.figure(figsize = (15, 4))

    plt.subplot(121)

    plt.plot(results[key]['accuracy'], label = 'acc')

    plt.plot(results[key]['val_accuracy'], label = 'val_acc')

    plt.legend()

    plt.title(f'{key} accuracy')

    

    plt.subplot(122)

    plt.plot(results[key]['loss'], label = 'loss')

    plt.plot(results[key]['val_loss'], label = 'val_loss')

    plt.legend()

    plt.title(f'{key} loss')

    

    plt.show()
for key in preds.keys():

    print(f'{key}: {accuracy_score(y_val, preds[key].argmax(axis = 1))}')
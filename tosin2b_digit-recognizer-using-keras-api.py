#lets import the required packages

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd

import tensorflow as tf
from keras import Sequential, Input, Model
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.layers import  Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam, RMSprop,Adadelta, Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import Softmax,LeakyReLU,activations
from sklearn.model_selection import train_test_split

#loading the datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#seperate the label into a variable y_train, and drop the column from our training set
y_train = train['label']
x_train = train.drop(['label'], axis =1) 
x_test = test
#Lets inspect the shape of the dataset
print('Train X  Dimension :',x_train.shape)
print('Test Dimension :',x_test.shape)
print(x_train.isnull().any().describe())
print(x_test.isnull().any().describe()) 
#Are the categories well represented ?
sns.set(style="darkgrid")
sns.countplot(y_train)
plt.xlabel(' Digits')
plt.ylabel('Count')
#Check for what categories the label contains
categories = y_train.unique()
print('Output Categories: ',categories)
print('Total number of  Categories: ',len(categories))
#categories
#x_train = (x_train.iloc[:,1:].values).astype('float32') # all pixel values
#y_train = y_train.values.astype('int32') # only labels i.e targets digits
#x_test = x_test.values.astype('float32')
X_train = x_train/255.0
X_test = x_test/255.0
x_train.shape
x_train = X_train.values.reshape(-1,28,28,1)
x_test = X_test.values.reshape(-1,28,28,1)
y_train_enc = to_categorical(y = y_train, num_classes= len(categories))
# Display the change for category label using one-hot encoding
print('Original label:', y_train[25])
print('After conversion to one-hot:', y_train_enc[25])
for i in range(0, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(x_train[i][:,:,0], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i]);
seed = 30
np.random.seed(seed)
#Spliting the Data
train_x, validation_x, train_y,validation_y = train_test_split(x_train,y_train_enc, test_size = 0.15, random_state = seed)
import keras
from keras import Sequential, Input, Model
from keras.layers import  Dense, Dropout, Flatten, Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import Softmax,LeakyReLU,activations
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
#some HyperParameters. Feel Free to tune them
batch_size = 128
epochs = 20
alpha = 0.3
num_classes = 10
#Model Architecture in Summary is [[Conv2D -> ReLU -> MaxPool2D -> DroupOut]] *2 -> Dense -> ReLU -> Flatten -> Droupout -> Dense -> Out
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='linear', padding='same',input_shape = (28,28,1)))
model.add(LeakyReLU(alpha=alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(rate = 0.5))

model.add(Conv2D(64, kernel_size=(3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=alpha))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(rate = 0.4))

model.add(Dense(128, activation='relu'))
model.add(LeakyReLU(alpha=alpha))
model.add(Flatten())

model.add(Dropout(rate=0.4))
model.add(Dense(len(categories), activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer=Adagrad(), metrics=['accuracy'])
model.summary()
#training the model
model_train = model.fit(x = train_x, y = train_y, batch_size= batch_size, epochs = epochs, validation_data=(validation_x,validation_y))
#extracting the training history params. this will give some information if its overfitting
train_acc = model_train.history['acc']
train_loss = model_train.history['loss']
val_acc = model_train.history['val_acc']
val_loss = model_train.history['val_loss']

ep = range(len(train_acc))
plt.plot(ep, train_acc,  label='Training accuracy', color ='g')
plt.plot(ep, val_acc, 'b', label='Validation accuracy',color='r')
plt.title('Training and validation accuracy')
plt.legend()

#plt.figure()
plt.plot(ep, train_loss, label='Training loss')
plt.plot(ep, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#Lets use our model to predict
y_pred = model.predict(validation_x,verbose =1)

#Convert predicted category to one hot vectors
y_pred_class = np.argmax(y_pred, axis =1 )

valid_y_class = np.argmax(validation_y, axis = 1)
#lets generate the confusion matrix so we can see how right the predictions are
confuse_matrix = confusion_matrix(valid_y_class,y_pred_class)

plt.figure(figsize = (10,10))
sns.heatmap(confuse_matrix, annot= True, fmt = 'd', cmap = 'YlGnBu', linewidths=.9,linecolor='black')
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(valid_y_class, y_pred_class, target_names=target_names))
errors = (y_pred_class - valid_y_class != 0)
y_pred_classes_errors = y_pred_class[errors]
y_pred_errors = y_pred[errors]
y_true_errors = valid_y_class[errors]
x_val_errors = validation_x[errors]
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
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, x_val_errors, y_pred_classes_errors, y_true_errors)
#Lets use the model to predict the testset from Kaggle Competition
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis = 1)
predictions = pd.Series(predictions, name =  'Label')
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),"Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
#training the model
model_train_full = model.fit(x = x_train, y = y_train_enc, batch_size= batch_size, epochs = epochs)
#Lets use the model to predict the testset from Kaggle Competition
predictions_full = model.predict(x_test)
predictions_full = np.argmax(predictions_full, axis = 1)
predictions_full = pd.Series(predictions_full, name =  'Label')
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions_full)+1)),"Label": predictions})
submissions.to_csv("Submission.csv", index=False, header=True)
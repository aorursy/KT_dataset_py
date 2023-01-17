import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 
print(os.listdir("../input"))
#load training and test data
train_data = pd.read_csv('../input/train.csv')
test_data =  pd.read_csv('../input/test.csv')
#Splilt features and labels
x = train_data[train_data.columns.difference(['label'])]
y = train_data[['label']]
validation_x  = test_data[test_data.columns.difference(['label'])] 
#Split the train set into train and test set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.30)
#We need to convert train_x and train_y into numpy arrays. As model only accepts numpy arrays
train_x = train_x.values
train_y = train_y.values
test_x  = test_x.values
test_y  = test_y.values
#Import Neural Net libraries
import tensorflow as tf
import tensorflow.keras as keras
#Normalize 
train_x = tf.keras.utils.normalize(train_x,axis=1)
test_x = tf.keras.utils.normalize(test_x,axis=1)

#Configure our model
network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Flatten())
network.add(keras.layers.Dense(128,activation=tf.nn.relu)) 
network.add(keras.layers.Dense(128,activation=tf.nn.relu)) # Relu works for most problem. 
network.add(keras.layers.Dense(10,activation=tf.nn.softmax)) #Softmax will generate the probability distributions of the output. The layer is predicting that the probability that the label is 1 is 0.10, label is 2 is 0.50 and so on. These all add upto 1. The number with the highest probability is treated as our models prediction. To fin the the number we just need to know the index of the number with higest probability (this is achieved by using numpy argmax function on the output)
#Compile the model with required loss, metrics and optimizer function
network.compile(metrics=['accuracy'], optimizer='adam',loss='sparse_categorical_crossentropy') 
# Categorical Cross Entropy is used for categorization problems. Sparse is used for numeric output.
# the loss function tells the model how to measure loss. The target for the optimizer then becomes to reduce the loss.
# Therefore, its essential that we choose to optimize the right loss value.
#fit the training data on the network
history = network.fit(train_x,train_y,epochs=15)
# Epochs simply means total number of iterations over test samples. Each iteraiton will cover all samples. There is an option of specifiying a batch size. If lets say we have 10,000 samples and we say batch size is 1000 then 10 batches will be processes in each epoch.
#history will store the accuracy and loss stats for each epoch. This would help us analyze the cost/benfit of running epochs. Accuracy will flatten after certain number of epochs and so would the loss. Thus additional epochs will only consume CPU and memory. so there is an oppurtunitiy to tune the parameters when compiling based on this analysis.
#Evaluate model
val_loss,val_acc = network.evaluate(test_x,test_y)

print("Loss on test data:")
print(val_loss)
print("Accuracy on test data (in %):")
print(val_acc*100)
# 96-97 % accuracy achieved by model. Main improvement was due to the introduction of data normalization and flatten layers in the network. Try removing these changes and see the negative impact on performance.
# The drop in accurach from training to test data reflects the fact that the model was overfitting (memorzing the results) on training data.
predictions  = network.predict(test_x)
idx = 23 #which test index do you want to try
print('\nProbability distribution for position :')
print(predictions[idx])
print('\nSum of probability scores :')
print(np.sum(predictions[idx]))
print('\nPredicted value :')

# The predicted values will be for the index which has the highest probability distribution
# argmax will check each value in the list and return the index values of the probability which has the higest probability value
# Which will also be the number predicted. Index values are - 0,1,2,3,4,5,6,7,8,9.
print(np.argmax(predictions[idx]))
print('\nActual value :')
print(test_y[idx][0])
#Lets analyze the numbers which our model is failing to predict properly. Is there a trend which we can work out and address the issue.

no_of_rows = test_x.shape[0]
incorrectly_predicted_number = [] #Stores the incorrected predicted numbers
for i in range(no_of_rows):
    predicted_value = np.argmax(predictions[i])
    actual_value = test_y[i][0]
    if predicted_value != actual_value:
        incorrectly_predicted_number.append(actual_value)
# Visually represent accuracy and loss over Epochs
import matplotlib.pyplot as plt

#Extract callback stats from loss and history.
loss = history.history['loss']
acc = history.history['acc']
epochs = range(1,len(loss)+1)

#Accuracy/Loss
plt.plot(epochs,acc,'bx',label='Accuracy stats')
plt.plot(epochs,loss,'bo',label='Loss stats')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train.columns
test.columns
Y_train = train['label']
X_train = train.drop(labels = ['label'], axis = 1 )
del train
Y_train.value_counts()
g = sns.countplot(Y_train)
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255.0
test = test/255.0
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)
g = plt.imshow(X_train[9][:,:,0])
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25)) #a proportion of nodes in the layer are randomly ignored


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
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print("Training...")
model.fit(X_train, Y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)
print("Generating test predictions...")
preds = model.predict_classes(test, verbose=0)
def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)
write_preds(preds, "keras-mlp.csv")

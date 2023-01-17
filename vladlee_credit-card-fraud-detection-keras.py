# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.head()
df.dropna()

df.drop(['Time','Amount'], axis=1, inplace=True)

df.describe()
import matplotlib.pyplot as plt



count_classes = pd.value_counts(df['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
number_records_fraud = len(df[df.Class == 1])

print(number_records_fraud)

fraud_indices = np.array(df[df.Class == 1].index)

normal_indices = df[df.Class == 0].index



# Out of the indices we picked, randomly select "x" number (number_records_fraud)

random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)

random_normal_indices = np.array(random_normal_indices)



# Appending the 2 indices

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])



# Under sample dataset

under_sample_data = df.iloc[under_sample_indices,:]



X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']

y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']



# Showing ratio

print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))

print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))

print("Total number of transactions in resampled data: ", len(under_sample_data))
from sklearn.manifold import TSNE



tsne2 = TSNE(n_components=2, perplexity=20, early_exaggeration=6)

X_tsne2 = tsne2.fit_transform(X_undersample)



# Plot the training points

x_min, x_max = X_tsne2[:, 0].min() - .5, X_tsne2[:, 0].max() + .5

y_min, y_max = X_tsne2[:, 1].min() - .5, X_tsne2[:, 1].max() + .5



fig = plt.figure(1, figsize=(8, 6))

col = y_undersample.values[:,0]

plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=col, cmap=plt.cm.Set1, edgecolor='k')

plt.show()
from sklearn.model_selection import train_test_split



# Undersampled dataset

X_train, X_test, y_train, y_test = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0,shuffle=True)



print("")

print("Number transactions train dataset: ", len(X_train))

print("Number transactions test dataset: ", len(X_test))

print("Total number of transactions: ", len(X_train)+len(X_test))
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras import optimizers

from keras.regularizers import l2 # L2-regularisation

from keras.callbacks import EarlyStopping

print(keras.__version__)
def create_model(units, optimizer, kernel_initializer):

    model = Sequential()

    model.add(Dense(units = units, activation = 'relu', input_dim = 28))

    model.add(Dropout(0.1))

    model.add(Dense(units = units, activation = 'relu'))

    model.add(Dropout(0.1))

    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
UNITS = 8

EPOCHS = 100

BATCH_SIZE = 32



model = create_model(UNITS, 'adam', 'glorot_uniform')

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = BATCH_SIZE, epochs = EPOCHS, 

                    verbose=1, shuffle=True, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='center right')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='center right')

plt.show()
df.drop(['Class'], axis=1, inplace=True)

predictions = model.predict(df.values)



#submitting the prediction results

sub_df=pd.DataFrame(data=predictions,columns=['predicted'])

#Save to csv file

sub_df.to_csv('submission.csv',index=False)
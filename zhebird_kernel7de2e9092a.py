# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import models

from keras import layers



from keras import optimizers

from keras import losses

from keras import metrics



import matplotlib.pyplot as plt



import math



pd.options.mode.chained_assignment = None





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")





train_data['BinarySex'] = float("NaN")

for index, temp in enumerate(train_data['Sex']):

    if (temp == 'male'):

        train_data['BinarySex'][index] = 0

    else:

        train_data['BinarySex'][index] = 1

ageSum = 0

ageCount = 0

for index, temp in enumerate(train_data['Age']):

    if (not math.isnan(temp)):

        ageSum = ageSum + temp

        ageCount = ageCount + 1

        

avgAge = ageSum / ageCount

for index, temp in enumerate(train_data['Age']):

    if (math.isnan(temp)):

        train_data['Age'][index] = avgAge

        



y_train = train_data['Survived'].to_numpy()

new_data = train_data.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'PassengerId', 'Survived'], axis=1)



print(new_data.head())



new_data = new_data.to_numpy()


def build_model():

    model = models.Sequential()

    model.add(layers.Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = new_data.shape[1]))

    model.add(layers.Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(layers.Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=['accuracy'])

    return model

k = 6

num_validation_samples = len(new_data) // k

np.random.shuffle(new_data)

histories = []



for fold in range(k):

    x_validation_data = new_data[num_validation_samples * fold:num_validation_samples * (fold + 1)]

    

   

    y_validation_data = y_train[num_validation_samples * fold:num_validation_samples * (fold + 1)]

    

    x_training_data = 0

    y_training_data = 0

    if fold != 0:

        x_training_data = np.vstack((new_data[:num_validation_samples * fold], new_data[num_validation_samples * (fold + 1):]))

        y_training_data = np.hstack((y_train[:num_validation_samples * fold], y_train[num_validation_samples * (fold + 1):]))

    else:

        x_training_data = new_data[num_validation_samples * (fold + 1):]

        y_training_data = y_train[num_validation_samples * (fold + 1):]

   

    

    model = build_model()

    

    history = model.fit(x_training_data, y_training_data, epochs=30, batch_size=10, validation_data=(x_validation_data, y_validation_data))

    histories.append(history)

   




def printHistory(history):

    history_dict = history.history

    loss_values = history_dict['loss']

    val_loss_values = history_dict['val_loss']



    print(loss_values)





    epochs = range(1, len(history_dict['binary_accuracy']) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')

    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

    plt.title('Training and validation loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()

    

for history in histories:

    printHistory(history)



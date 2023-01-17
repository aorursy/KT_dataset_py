import pandas as pd

import numpy as np
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
target = train['label']

train = train.drop('label', axis = 1)
print(target.shape)
print(train.shape)
print(test.shape)
train.head()
test.head()
train.describe()
train.isnull().count()
target
train.values[0]
from keras.models import Sequential

from keras.layers import Dense
# Initialising ANN

classifier = Sequential()



# Adding 5 hidden layers

for i in range(4):

    if(i == 0):

        classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu', input_dim = 784))

    else:

        classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))

        

# Adding output layer

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid', input_dim = 5))



# Compiling ANN

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# training ANN on train dataset

classifier.fit(train, target, batch_size = 20 , epochs = 15)
test_pred = classifier.predict(test)
test_pred
pred_int = []

for i in range(test_pred.shape[0]):

    pred_ratios = test_pred[i]

    for j in range(10):

        if(pred_ratios[j] == max(pred_ratios)):

            number = int(j)

    pred_int = np.append(pred_int, number)
pred_int = pred_int.astype(int)
submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
submission['Label'] = pred_int

submission
submission.to_csv('submission.csv', index = False)
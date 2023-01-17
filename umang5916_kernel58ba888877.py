# Importing the nessecary Libraries for the program

import pandas as pd

import numpy as np
# Importing the training dataset

dataset = pd.read_csv('../input/zaloni-techniche-datathon-2019/train.csv')

X = dataset.iloc[:, 0:2].values

Y = dataset.iloc[:, 2:4].values

X = X.astype(str)
# Encoding all the possible outcomes(ie. 8)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

Y[:, 0] = labelencoder.fit_transform(Y[:, 0])      # encodes male, female as 1, 0

Y[:, 1] = labelencoder.fit_transform(Y[:, 1])      # encodes all the race as 0, 1, 2, 3

Y = Y.astype(float)

Y[:, 0] = Y[:, 0]*10+Y[:, 1]                       # 8 possible outcomes are 0,1,2,3,10,11,12,13

Y[:, 0] = labelencoder.fit_transform(Y[:, 0])

Y = np.delete(Y, 1, axis = 1)
# counting the number of times an alphabet or space has occured in first name and last name

for i in range(0,54):                      # adding extra column for the counts of alphabets and the spaces

    X = np.c_[X, np.zeros(85272)]          # 54 = 26(alphabet)*2(first and last name)+2(space for each first and last name)



for j in range(0,85272):

    Z = X[j,0]

    for k in range(97,123):

        count = 0

        count = Z.count(chr(k))           # counting alphabets in the last name

        X[j, 3+k-97] = count              # saving the counts in column 3 to 28

for j in range(0,85272):

    Z = X[j,1]

    for k in range(97,123):

        count = 0

        count = Z.count(chr(k))          # counting aplhabets in the first name

        X[j, 29+k-97] = count            # saving the counts in column 29 to 54

for j in range(0,85272):

    count = 0

    Z = X[j, 0]

    count = Z.count(' ')                 # counting the number of spaces in the last name and saving it in the column 2

    X[j, 2] = count

for j in range(0,85272):

    count = 0

    Z = X[j, 1]

    count = Z.count(' ')                 # counting the number of spaces in the first name and saving it in the column 55

    X[j, 55] = count

    

X = np.delete(X, 0, axis = 1)            # removing the last name column

X = np.delete(X, 0, axis = 1)            # removing the first name column

X = X.astype(float)
# importing the test dataset

dataset2 = pd.read_csv('../input/zaloni-techniche-datathon-2019/test.csv')

test = dataset2.iloc[:, [1,2]].values

test = test.astype(str)
# counting the number of times an alphabet or space has occured in first name and last name (same as we did for the training set)

for i in range(0,54):

    test = np.c_[test, np.zeros(12186)]



for j in range(0,12186):

    Z = test[j,0]

    for k in range(97,123):

        count = 0

        count = Z.count(chr(k))

        test[j, 3+k-97] = count

for j in range(0,12186):

    Z = test[j,1]

    for k in range(97,123):

        count = 0

        count = Z.count(chr(k))

        test[j, 29+k-97] = count

for j in range(0,12186):

    count = 0

    Z = test[j, 0]

    count = Z.count(' ')

    test[j, 2] = count

for j in range(0,12186):

    count = 0

    Z = test[j, 1]

    count = Z.count(' ')

    test[j, 55] = count

    

test = np.delete(test, 0, axis = 1)

test = np.delete(test, 0, axis = 1)

test = test.astype(float)
# fitting the Random Forest Classifier to the training dataset

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0)

classifier.fit(X, Y)
# predicting the outcome for test dataset

Y_pred_1 = classifier.predict(test)
# converting the predicted dataset to gender and race

Y_pred_1 = labelencoder.inverse_transform(Y_pred_1.astype(int))

Y_pred_1 = np.c_[Y_pred_1, np.zeros(12186)]

Y_pred_1[:, 1] = Y_pred_1[:,0]%10

Y_pred_1[:, 0] = (Y_pred_1[:, 0]/10).astype(int)



Y_pred_1 = Y_pred_1.astype(object)



for i in range(0, 12186):

    if( Y_pred_1[i, 0] == 0):

        Y_pred_1[i, 0] = 'f'

    else:

        Y_pred_1[i, 0] = 'm'

for i in range(0, 12186):

    if( Y_pred_1[i, 1] == 0 ):

        Y_pred_1[i, 1] = 'black'

    elif( Y_pred_1[i, 1] == 1 ):

        Y_pred_1[i, 1] = 'hispanic'

    elif( Y_pred_1[i, 1] == 2 ):

        Y_pred_1[i, 1] = 'indian'

    elif( Y_pred_1[i, 1] == 3 ):

        Y_pred_1[i, 1] = 'white'



Y_pred_1 = Y_pred_1.astype(str)
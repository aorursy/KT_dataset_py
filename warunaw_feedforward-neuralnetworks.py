import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from numpy import loadtxt

from keras.models import Sequential

from keras.layers import Dense



import pandas as pd 

from sklearn.metrics import f1_score
dataset = pd.read_csv('/kaggle/input/creditscreening/credit-screening.data')



dataset.head()
colNames = []



for i in range(15):

    x = "A" + str(i+1)

    colNames.append(x)



colNames.append('class')

dataset.columns = colNames

dataset.tail()
dataset.isna().sum()
dataset.dtypes
dataset.replace('?', np.nan, inplace=True)

dataset.isna().sum()
dataset = dataset.fillna(method ='pad')

dataset.isna().sum()
dataset['A14'] = dataset['A14'].astype('int64')

dataset['A2'] = dataset['A2'].astype('float64')

dataset.dtypes
dataset['A1'] = dataset['A1'].astype('category')

dataset['A4'] = dataset['A4'].astype('category')

dataset['A5'] = dataset['A5'].astype('category')

dataset['A6'] = dataset['A6'].astype('category')

dataset['A7'] = dataset['A7'].astype('category')

dataset['A9'] = dataset['A9'].astype('category')

dataset['A10'] = dataset['A10'].astype('category')

dataset['A12'] = dataset['A12'].astype('category')

dataset['A13'] = dataset['A13'].astype('category')

print(dataset.info())
dataset['A1'] = dataset['A1'].cat.codes

dataset['A4'] = dataset['A4'].cat.codes

dataset['A5'] = dataset['A5'].cat.codes

dataset['A6'] = dataset['A6'].cat.codes

dataset['A7'] = dataset['A7'].cat.codes

dataset['A9'] = dataset['A9'].cat.codes

dataset['A10'] = dataset['A10'].cat.codes

dataset['A12'] = dataset['A12'].cat.codes

dataset['A13'] = dataset['A13'].cat.codes
dataset.head()
np.random.seed(1337)
dataset.to_csv('credit-screening-all-numerics.csv', index=None)
from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



skf = StratifiedKFold(n_splits=5,random_state=42, shuffle=False)



X = dataset.iloc[:,:-1]

Y = dataset.iloc[:,-1]



Y.replace('+', 1, inplace=True)

Y.replace('-', 0, inplace=True)



print(len(X.columns))
X = (X-X.min())/(X.max()-X.min())

print(X.head())
from keras import backend as K

import matplotlib.pyplot as plt
def doTrainAndEvaluation(hiddenlayerNodeCount, hiddenLayerActivation, outputLayerActivation, lossFunction, plot_history=False):

    print("DoTrainAndEvaluation with hidden nodes=%s" %hiddenlayerNodeCount)

    

    scores = []

    fold = 0        

    

    for train_index, test_index in skf.split(X, Y):

        fold = fold + 1

        x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

        

        model = Sequential()

        model.add(Dense(hiddenlayerNodeCount, input_dim=15, activation=hiddenLayerActivation, kernel_initializer='normal'))

        model.add(Dense(1, activation=outputLayerActivation, kernel_initializer='normal'))

        model.compile(loss=lossFunction, optimizer='adam', metrics=['acc'])

        

        history = model.fit(x_train, y_train, epochs=50, verbose=0, batch_size=100, validation_split=0.2)

        

        if plot_history == True:

            print("###### Cross validation fold number = %s" %fold)

            plt.plot(history.history['acc'])

            plt.plot(history.history['loss'])

            plt.plot(history.history['val_loss'])

            plt.title('Model loss')

            plt.ylabel('Loss')

            plt.xlabel('Epoch')

            plt.legend(['accuracy','Train', 'Valildation'], loc='upper right')

            plt.show()

        

        y_pred = model.predict(x_test, verbose=0)

        y_pred = np.where(y_pred > 0.5, 1, 0)

        f1 = f1_score(y_test, y_pred, average='macro')

        scores.append(f1)



    print("Hidden layer count={} Mean values for f1={}".format(hiddenlayerNodeCount, np.mean(scores, axis=0)))

    print("========================================Model complete========================================")

    

    return np.mean(scores, axis=0)
import matplotlib.pylab as plt



def plot_summary(result_dict, s = "hidden layer neuron count"):      

    items = result_dict.items()

    x,y = zip(*items)

    plt.plot(x, y)

    plt.xlabel(s)

    plt.ylabel('F1 value')

    maximum_f1_value = max(y)

    hidden_neuron_count = max(result_dict, key=lambda k: result_dict[k])



    print("Maximum f1 val=" + str(maximum_f1_value) + ", " + s + "=" +  str(hidden_neuron_count))
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'sigmoid', 'sigmoid', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'sigmoid', 'sigmoid', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'relu', 'relu', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'relu', 'relu', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'tanh', 'tanh', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'tanh', 'tanh', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'linear', 'linear', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'linear', 'linear', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'sigmoid', 'tanh', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'sigmoid', 'tanh', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'tanh', 'sigmoid', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluation(i, 'tanh', 'sigmoid', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
def doTrainAndEvaluationTwoHiddenLayers(hidL1NodeCount, hidL1Activation,hidL2NodeCount, hidL2Activation, outputLActivation, lossFunction, plot_history=False):

    print("DoTrainAndEvaluation with hidden Layer 1 nodes={} Hidden layer2 nodes={}".format(hidL1NodeCount,hidL2NodeCount))

    

    scores = []

    fold = 0        

    

    for train_index, test_index in skf.split(X, Y):

        fold = fold+1

        x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

        

        model = Sequential()

        model.add(Dense(hidL1NodeCount, input_dim=15, activation=hidL1Activation, kernel_initializer='normal'))

        model.add(Dense(hidL2NodeCount, activation=hidL2Activation, kernel_initializer='normal'))

        model.add(Dense(1, activation=outputLActivation, kernel_initializer='normal'))

        model.compile(loss=lossFunction, optimizer='adam', metrics=['acc'])

        

        history = model.fit(x_train, y_train, epochs=50, verbose=0, batch_size=100, validation_split=0.2)

        

        if plot_history == True:

            print("###### Cross validation fold number = %s" %fold)

            plt.plot(history.history['acc'])

            plt.plot(history.history['loss'])

            plt.plot(history.history['val_loss'])

            plt.title('History plot on Model loss')

            plt.ylabel('loss')

            plt.xlabel('Epoch')

            plt.legend(['Acc','Train', 'Test'], loc='upper left')

            plt.show()

        

        y_pred = model.predict(x_test, verbose=0)

        y_pred = np.where(y_pred > 0.5, 1, 0)

        f1 = f1_score(y_test, y_pred, average='macro')

        scores.append(f1)



    print("Mean values for f1={}".format(np.mean(scores, axis=0)))

    print("========================================Model complete========================================")

    

    return np.mean(scores, axis=0)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluationTwoHiddenLayers(i, 'sigmoid', i, 'sigmoid', 'sigmoid', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluationTwoHiddenLayers(i, 'tanh', i, 'tanh', 'tanh', 'mean_squared_error', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluationTwoHiddenLayers(i, 'tanh', i, 'tanh', 'tanh', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
matrices_variation = dict()



for i in range(1, 16, 1):

    matrices_variation[i] = doTrainAndEvaluationTwoHiddenLayers(i, 'sigmoid', i, 'tanh', 'tanh', 'binary_crossentropy', plot_history=False)
plot_summary(matrices_variation)
from keras import regularizers

import numpy

from sklearn.model_selection import train_test_split

from collections import defaultdict 



def addRegularization(regularization, showHistory=False):

    variation = defaultdict(list)

    scores = []

    fold = 0        

    for train_index, test_index in skf.split(X, Y):

        fold = fold+1

        x_train, x_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]



        for i in numpy.arange(0, 0.01, 0.001): 

            model = Sequential()

            if (regularization == "l1"):

                model.add(Dense(14, input_dim=15, activation='linear', kernel_regularizer=regularizers.l1(i)))

                model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l1(i)))

            else:

                model.add(Dense(14, input_dim=15, activation='linear', kernel_regularizer=regularizers.l2(i)))

                model.add(Dense(1, activation='linear', kernel_regularizer=regularizers.l2(i)))

                

            model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

            history = model.fit(x_train, y_train, epochs=50, verbose=0, validation_split=0.2)



            if showHistory == True:

                print("###### Cross validation fold number = %s" %fold)

                plt.plot(history.history['acc'])

                plt.plot(history.history['val_loss'])

                plt.plot(history.history['loss'])

                plt.title('Model loss and accuracy')

                plt.ylabel('loss and accuracy')

                plt.xlabel('Epoch')

                plt.legend(['accuracy', 'validation loss', 'loss'], loc='upper right')

                plt.show()



            y_pred = model.predict(x_test, verbose=0)

            y_pred = np.where(y_pred > 0.5, 1, 0)

            f1 = f1_score(y_test, y_pred, average='macro')

            scores.append(f1)

           

            variation[i].append(f1)



    return variation
matrices_variation = addRegularization('l1', False)

print(matrices_variation)
avg_f1_perfl1 = dict()



for key, val in matrices_variation.items():

    avg_f1_perfl1[key] = np.mean(val)

    

print(avg_f1_perfl1)

plot_summary(avg_f1_perfl1,"F1 value variation with different lambda in L1")
matrices_variation = addRegularization('l2')

print(matrices_variation)
avg_f1_perfl2 = dict()



for key, val in matrices_variation.items():

    avg_f1_perfl2[key] = np.mean(val)

    

print(avg_f1_perfl2)

plot_summary(avg_f1_perfl2, "F1 value variation with different lambda in L2")
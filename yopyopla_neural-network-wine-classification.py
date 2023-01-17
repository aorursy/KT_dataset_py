import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam, rmsprop, sgd
from keras.callbacks import EarlyStopping
import keras.backend as Ka
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import KFold
from numpy import array
from keras import optimizers
# importing data
database = np.genfromtxt("../input/Wine.csv", delimiter=",", dtype=float)
database[[0,4,20,96],:]

# to have categories ( 0, 1 and 2 rathen than 1,2 and 3)
database[:,0] = database[:,0] - 1
database[:,0]

## One hot encode the columns of the database specified in list_id.
#
#  return a copy of the database, on hot encoded. 
def one_hot_encode_database(database, list_id):
    encoded_database = np.empty(shape=(database.shape[0],0), dtype=float)
    for id in range(database.shape[1]):
        if id in list_id:
            original_column = database[:, id]
            encoded_column = to_categorical(original_column)
            encoded_database = np.column_stack((encoded_column,encoded_database))
        else:
            original_column = database[:, id]
            encoded_database = np.column_stack((original_column,encoded_database))
    return encoded_database

## Normalize between 0 and 1 each column of the database specified in list_id.
#
#  return a copy of the database, normalized. 
def normalize_database(database, list_id):
    encoded_database = database.copy()
    for  id in list_id :
        x = encoded_database[:, id]
        encoded_database[:, id] = (x-min(x))/(max(x)-min(x))
    return encoded_database

database = one_hot_encode_database(database,[0])
database = normalize_database(database, range(12))
database.shape[0]
predictors = database[:,:-3]
target = database[:,-3:]

## Split the input data into nb_fold equal folds. 
#     Select current_fold as the test_dataset and the stack the other folds to be the train_dataset.
#
#  return a training_dataset and a test_dataset.
def get_kfold_cv(input_data, nb_fold, current_fold):
    kfold = KFold(nb_fold, True, 1)
    index = range(input_data.shape[0])
    i_fold = 0
    for train, test in kfold.split(index):
        if current_fold == i_fold:
            return(input_data[train,:], input_data[test,:])
        i_fold = i_fold + 1
get_kfold_cv(database, 5, 1)[1][1]
## Construct a layer composed of dense layers, which dimensions are definded in the layer_list argument.
# 
#  return the constructed and compiled model.
def build_NN(layer_list, input_dim, output_dim, lr=0.001):
    model = Sequential();
    
    for idx, layer in enumerate(layer_list):
        if(idx == 0):
            model.add(Dense(layer,activation='relu',input_shape=(input_dim,)))
        else:
            model.add(Dense(layer,activation='relu'))

    model.add(Dense(output_dim,activation='softmax'))
    
    sgd = optimizers.SGD(lr=lr)
    #Compile the network
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def test_model(database,lr,epochs = 100, layer_list = [50,50,20],one_loop=False):
    np.random.shuffle(database)

    nb_fold = 5
    gt_and_pred = np.empty(shape=(2,0)) 

    for fold in range(nb_fold):
        K.clear_session()
        model = build_NN(layer_list, database.shape[1]-3, 3, lr=lr)
        model.summary()
        training_fold, valid_fold = get_kfold_cv(database, nb_fold, fold)
        callback = EarlyStopping(patience=500)
        model.fit(training_fold[:,:-3], training_fold[:,-3:], batch_size=16, epochs=epochs, validation_data=(valid_fold[:,:-3], valid_fold[:,-3:]), shuffle=True, callbacks=[callback])
        fold_prediction = model.predict(valid_fold[:,:-3])

        gt_and_pred = np.column_stack((gt_and_pred, np.array([np.argmax(valid_fold[:,-3:], axis=1), np.argmax(fold_prediction, axis=1)])))
        if one_loop:
            break
    kappa = cohen_kappa_score(gt_and_pred[0], gt_and_pred[1], labels=[0,1,2])
    conf_mat = confusion_matrix(gt_and_pred[0], gt_and_pred[1], labels=[0,1,2])
    return(kappa,conf_mat)
%%time
model1 = test_model(database,0.001,7000,[5000],False)
print("Cohen's kapp : ")
print(model1[0])
print("")
print("Confusion matrix : ")
print(model1[1])
print("")
confu_matrix = model1[1]
print('Accuracy : ' + str(np.diag(confu_matrix).sum()/confu_matrix.sum()*100)  )

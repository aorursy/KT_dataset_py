import warnings  

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

from datetime import datetime, date, time, timedelta

import math

import json

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import recall_score, precision_score, roc_auc_score

from keras.models import Sequential

from keras.layers import LSTM, Dropout, Dense, InputLayer

from keras.callbacks import EarlyStopping

from keras import optimizers

from keras import backend as K

import h5py

import math
%cd input

%ls
#fichero='query (14).csv' #USA

fichero='/kaggle/input/query20190601.csv'

#fichero='queryCuadrado.csv'

earthquake =pd.read_csv(fichero, sep=',')

earthquake.head(1)
print ("número de terremotos:", earthquake.count()[0])
print('Fecha inicial: ', earthquake['time'].min())

print('Fecha final: ', earthquake['time'].max())
earthquake['latitude'].describe()
earthquake = earthquake.sort_values(by='time').reindex()
plt.figure(figsize=(10,3))

earthquake["mag"].hist() ##bins=4

plt.title('Magnitud de terremotos')
earthquake["mag"].describe()
# Separamos la variable "time" (YYYY-MM-DDThh:mm:ss.s) en "date" en formato YYYY-MM-DD

# y "time" sobreescrita en formato hh:mm:ss.s

earthquake["date"] = earthquake["time"].apply(lambda x: x.split("T")[0])

earthquake["time"] = earthquake["time"].apply(lambda x: x.split("Z")[0].split("T")[1])

earthquake[["date","time"]].head(1)
#Vnw, Vsw, Vne, Vse

la_min = earthquake["latitude"].min()

print("Minimum Latitude: ",la_min)

la_max = earthquake["latitude"].max()

print("Maximum Latitude: ",la_max)

lo_min = earthquake["longitude"].min()

print("Minimum Longitude: ",lo_min)

lo_max = earthquake["longitude"].max()

print("Maximum Longitude: ",lo_max)
#Vectores de posición que limitan el área de estudio: Vnw, Vsw, Vne, Vse

Vnw = (lo_min, la_max) #Northwest

Vsw = (lo_min, la_min) #Southwest

Vne = (lo_max, la_max) #Northeast

Vse = (lo_max, la_min) #Southeast

#Vnw, Vsw, Vne, Vse

print('Vnw: ',Vnw)

print('Vsw: ',Vsw)

print('Vne: ',Vne)

print('Vse: ',Vse)

# Número de subdivisiones por base y altura:

mb = 3

ma = 3

#Base y altura de las subáreas de estudio:

b = round(abs(lo_max - lo_min)/mb,4)

a = round(abs(la_max - la_min)/ma,4)

print("Altura: ", a, "Base: ", b)
earthquake["i"] = earthquake["latitude"].apply(lambda x: (x-la_min)/a)

earthquake["j"] = earthquake["longitude"].apply(lambda x: (x-lo_min)/b)

        

# earthquake["int_i"] = np.ceil(earthquake["i"])

# earthquake["int_j"] = np.ceil(earthquake["j"])
#plt.plot(earthquake["int_i"],earthquake["int_j"])

plt.figure(figsize=(20,20))

# plt.scatter(earthquake["int_i"],earthquake["int_j"])

plt.scatter(earthquake["i"],earthquake["j"])

plt.grid(True)

plt.show()
earthquake["k"] = 0.0

earthquake.loc[(earthquake.i < 1.0) & (earthquake.j < 1.0), 'k'] = 1.0

earthquake.loc[(earthquake.i < 1.0) & (1.0 <= earthquake.j) & (earthquake.j < 2.0), 'k'] = 2.0

earthquake.loc[(earthquake.i < 1.0) & (2.0 <= earthquake.j) & (earthquake.j < 3.0), 'k'] = 3.0

earthquake.loc[(1.0 <= earthquake.i) & (earthquake.i < 2.0) & (earthquake.j < 1.0), 'k'] = 4.0

earthquake.loc[(1.0 <= earthquake.i) & (earthquake.i < 2.0) & (1.0 <=earthquake.j) & (earthquake.j< 2.0), 'k'] = 5.0

earthquake.loc[(1.0 <= earthquake.i) & (earthquake.i < 2.0) & (2.0 <=earthquake.j) & (earthquake.j < 3.0), 'k'] = 6.0

earthquake.loc[(2.0 <= earthquake.i) & (earthquake.i < 3.0) & (earthquake.j < 1.0), 'k'] = 7.0

earthquake.loc[(2.0 <= earthquake.i) & (earthquake.i < 3.0) & (1.0 <=earthquake.j) & (earthquake.j < 2.0), 'k'] = 8.0

earthquake.loc[(2.0 <= earthquake.i) & (earthquake.i < 3.0) & (2.0 <=earthquake.j) & (earthquake.j <3.0), 'k'] = 9.0

earthquake
max_date = datetime.strptime(earthquake["date"].max(), '%Y-%m-%d')

min_date = datetime.strptime(earthquake["date"].min(), '%Y-%m-%d')

print(max_date)

print(min_date)
def diff_month(d1, d2):

    return np.ceil((d1 - d2).days / 14)
earthquake["quincena"] = earthquake.date.apply(lambda d: diff_month(datetime.strptime(d, '%Y-%m-%d'), min_date))
earthquake
vectors = pd.DataFrame()

vectors = pd.crosstab(earthquake["k"], earthquake["quincena"]) #Cálculo de frecuencias

vectors
d = pd.DataFrame(np.zeros((1,int(earthquake["quincena"].max()))))

vectors = vectors.append(d)

vectors =  vectors.fillna(0.0)

vectors = vectors.T

vectors = vectors.iloc[:,1:10]

vectors
region = vectors.iloc[:,4]
print('queryCuadrado.csv - 0.0',region[region == 0.0].count())

print('queryCuadrado.csv - 1.0',region[region == 1.0].count())
porcentaje_no_terremotos = region[region == 0.0].count() * 100 / region.count()

print('porcentaje de terremotos: {:.4f}'.format(100 - porcentaje_no_terremotos))
hoy = datetime.today().strftime('%Y%m%d')

# nombreFichero = 'vectors_Lorca3x3-{}.nb.csv'.format(hoy)

# #nombreFichero = 'vectors_mediterraneo3x3-20190608.csv'

# vectors.to_csv(path_or_buf=nombreFichero, index=False)
vectors2 = vectors

pd.DataFrame(vectors2.iloc[879:902,4].apply(lambda x: 1 if x>0 else 0))
zona = 4

region = vectors2.iloc[:,zona]

region.describe()
metrics = dict()
region_central = vectors2.iloc[:,zona]

train_size = int(len(region_central) * 0.7)

train = region_central[0:train_size]

test_size = int(len(region_central) * 0.1) + train_size

test = region_central[train_size:test_size]

val = region_central[test_size:len(region_central)]
print('train_size: ', len(train))

print('test_size: ', len(test))

print('val_size: ', len(val))





len(train) + len(test) + len(val) == len(region_central)
def create_dataset(dataset, window_size = 1):

    data_X, data_Y = [], []

    #scaler = MinMaxScaler()

    for i in range(len(dataset) - window_size - 1):

        a = dataset.values[i:i + window_size]

        data_X.append(a)       

        data_Y.append(dataset.iloc[i + window_size])

    X = np.array(data_X)

    #X = scaler.fit_transform(X)

    Y = np.array(data_Y)

    Y[Y > 1] = 1

    return X,Y
def DrawGraphcs(model):

    plt.figure(figsize=(20,5))

    plt.subplot(1, 2, 1)

    plt.plot(model.history.history['accuracy'])

    plt.plot(model.history.history['val_accuracy'])

    plt.subplot(1, 2, 2)

    plt.plot(model.history.history['loss'])

    plt.plot(model.history.history['val_loss'])
def Metrics_LR(train_X, train_Y, test_X, test_Y, name, window_size):

    lr_train_X, lr_train_Y = train_X, train_Y

    lr_test_X, lr_test_Y = test_X, test_Y



    clf = LogisticRegression()

    clf.fit(lr_train_X, lr_train_Y)



    lr_y_pred = clf.predict(lr_test_X)



    acc = clf.score(lr_test_X, lr_test_Y)

    print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(acc))

    precision = precision_score(lr_test_Y, lr_y_pred)

    print('Precision of logistic regression classifier on test set: {:.4f}'.format(precision))

    recall = recall_score(lr_test_Y, lr_y_pred)

    print('Recall of logistic regression classifier on test set: {:.4f}'.format(recall))

    roc = roc_auc_score(lr_test_Y, lr_y_pred)

    print('Roc of logistic regression classifier on test set: {:.4f}'.format(roc))

    print('Training data shape:{}'.format(train_X.shape))

    print('Testing data shape:{}'.format(test_X.shape))

    metrics[name] = [window_size, train_X.shape, test_X.shape, acc, precision, recall, roc]
def Metrics_NN(model, model_name, test_X, test_Y, window_size):

    y_pred = model.predict(test_X).round()

    score, acc = model.evaluate(test_X, test_Y, batch_size=window_size)

    print('score: ', score)

    print('acc: ', acc)

    precision = precision_score(test_Y, y_pred)

    print('Precision: {:.4f}'.format(precision))

    recall = recall_score(test_Y, y_pred)

    print('Recall: {:.4f}'.format(recall))

    roc = roc_auc_score(test_Y, y_pred)

    print('ROC: {:.4f}'.format(roc))

    print('Training data shape:{}'.format(train_X.shape))

    print('Testing data shape:{}'.format(test_X.shape))

    metrics[model_name] = [model.count_params(), train_X.shape, test_X.shape, acc, precision, recall, roc]
def Save_model(model, model_name):

    hoy = datetime.today().strftime('%Y%m%d')

    model.name = model_name

    model.save('{}_{}.h5'.format(model.name,hoy))

    with open('{}_{}_history.json'.format(model.name,hoy), 'w') as f:

        json.dump(model.history.history, f)
# Create test and training sets for one-step-ahead regression.

window_size = 1

train_X, train_Y = create_dataset(train, window_size)

test_X, test_Y = create_dataset(test, window_size)

val_X, val_Y = create_dataset(val, window_size)



print("Original training data shape:")

print(train_X.shape)

Metrics_LR(train_X, train_Y, test_X, test_Y, "lr_1", window_size)
def fit_model_mlp (X_train, y_train, window_size = 1, X_val = None, y_val = None, val = False):

    model = Sequential()

    model.add(Dense(4, bias_initializer='ones', input_dim=window_size, activation='tanh', name = 'Dense6'))

    model.add(Dense(1, bias_initializer = 'ones', activation='sigmoid', name = 'DenseOutput'))

    model.compile(loss = "binary_crossentropy", 

                  optimizer = "adam", metrics=["accuracy"])

    

    print(model.summary())

    

    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    

    if val:

            model.fit(X_train, 

              y_train, 

              epochs = 100, 

              batch_size = window_size, 

              verbose = 2,

              validation_data=(X_val, y_val),

              callbacks = [earlystopper])

    else:

        model.fit(X_train, 

                  y_train, 

                  epochs = 100, 

                  batch_size = window_size, 

                  verbose = 1,

                  callbacks = [earlystopper])

    

    #print(model.summary())

    return(model)

    

    
mlp1 = fit_model_mlp(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(mlp1)
Metrics_NN(mlp1, "mlp_1", test_X, test_Y, window_size)
models = dict()

models["mlp1"] = mlp1
# Reshape the input data into appropriate form for Keras.

train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

val_X = np.reshape(val_X, (val_X.shape[0], 1, val_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

print("New training data shape:")

print(train_X.shape)

print(train_X[0])
def fit_model(X_train, y_train, window_size = 1, X_val = None, y_val = None, val = False):

    model = Sequential()

    

#     model.add(LSTM(1,

#                    bias_initializer='ones',

#                    input_shape = (1, window_size),

#                    return_sequences=True,

#                    name='InputLayer')

#                    )

    

    model.add(LSTM(units=4, return_sequences=True, bias_initializer='ones', input_shape = (1, window_size), name = 'HiddenLayer1'))

    #model.add(Dropout(0.33))

    model.add(LSTM(units=2, return_sequences=True, bias_initializer='ones', name = 'HiddenLayer2'))

    #model.add(Dropout(0.33))

    model.add(LSTM(units=1, activation='softmax', bias_initializer='ones', name='OutputLayer'))

    #model.add(Dense(1))

    model.compile(loss = "binary_crossentropy", 

                  optimizer = "adam", metrics=["accuracy"])

    print(model.summary())

    

    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    

    if val:

            model.fit(X_train, 

              y_train, 

              epochs = 100, 

              batch_size = window_size, 

              verbose = 2,

              validation_data=(X_val, y_val),

              callbacks = [earlystopper])

    else:

        model.fit(X_train, 

                  y_train, 

                  epochs = 100, 

                  batch_size = window_size, 

                  verbose = 1,

                  callbacks = [earlystopper])

    

    return(model)
# Fit the first model.

#model1 = fit_model(train_X, train_Y, window_size)

lstm1 = fit_model(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(lstm1)
Metrics_NN(lstm1, "lstm_1", test_X, test_Y, window_size)
metrics
window_size = 2

train_X, train_Y = create_dataset(train, window_size)

test_X, test_Y = create_dataset(test, window_size)

val_X, val_Y = create_dataset(val, window_size)
Metrics_LR(train_X, train_Y, test_X, test_Y, "lr_2", window_size)
mlp2 = fit_model_mlp(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(mlp2)
Metrics_NN(mlp2, "mlp_2", test_X, test_Y, window_size)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

val_X = np.reshape(val_X, (val_X.shape[0], 1, val_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
lstm2 = fit_model(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(lstm2)
Metrics_NN(lstm2, "lstm_2", test_X, test_Y, window_size)
metrics
window_size = 3

train_X, train_Y = create_dataset(train, window_size)

test_X, test_Y = create_dataset(test, window_size)

val_X, val_Y = create_dataset(val, window_size)



Metrics_LR(train_X, train_Y, test_X, test_Y, "lr_3", window_size)
mlp3 = fit_model_mlp(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(mlp3)
Metrics_NN(mlp3, "mlp_3", test_X, test_Y, window_size)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

val_X = np.reshape(val_X, (val_X.shape[0], 1, val_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
lstm3 = fit_model(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(lstm3)
Metrics_NN(lstm3, "lstm_3", test_X, test_Y, window_size)
metrics
window_size = 4

train_X, train_Y = create_dataset(train, window_size)

test_X, test_Y = create_dataset(test, window_size)

val_X, val_Y = create_dataset(val, window_size)
Metrics_LR(train_X, train_Y, test_X, test_Y, "lr_4", window_size)
mlp4 = fit_model_mlp(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(mlp4)
Metrics_NN(mlp4, "mlp_4", test_X, test_Y, window_size)
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))

val_X = np.reshape(val_X, (val_X.shape[0], 1, val_X.shape[1]))

test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
lstm4 = fit_model(train_X, train_Y, window_size, val_X, val_Y, True)
DrawGraphcs(lstm4)
Metrics_NN(lstm4, "lstm_4", test_X, test_Y, window_size)
metrics
region_total = vectors2

train_size = int(len(region_total) * 0.7)

train9 = region_total[0:train_size]

test_size = int(len(region_total) * 0.1) + train_size

test9 = region_total[train_size:test_size]

val9 = region_total[test_size:len(region_total)]
def create_dataset(dataset, window_size = 1):

    data_X, data_Y = [], []

    for i in range(len(dataset) - window_size - 1):

        a = dataset.values[i:i + window_size,:]

        data_X.append(a)

        data_Y.append(dataset.iloc[i + window_size,4])

    X = np.array(data_X)

    Y = np.array(data_Y)

    Y[Y>1] = 1

    return X,Y
def Metrics_LR_9(train_X, train_Y, test_X, test_Y, name, window_size):

    lr_train_X, lr_train_Y = train_X, train_Y

    lr_test_X, lr_test_Y = test_X, test_Y

    

    lr_train_X9 = np.reshape(train_X, (train_X.shape[0], train_X.shape[1] * 9))

    lr_train_Y9 = train_Y

    lr_test_X9 = np.reshape(test_X, (test_X.shape[0], test_X.shape[1] * 9))

    lr_test_Y9 = test_Y



    clf = LogisticRegression()

    clf.fit(lr_train_X9, lr_train_Y9)



    lr_y_pred = clf.predict(lr_test_X9)



    acc = clf.score(lr_test_X9, lr_test_Y9)

    print('Accuracy of logistic regression classifier on test set: {:.4f}'.format(acc))

    precision = precision_score(lr_test_Y9, lr_y_pred)

    print('Precision of logistic regression classifier on test set: {:.4f}'.format(precision))

    recall = recall_score(lr_test_Y9, lr_y_pred)

    print('Recall of logistic regression classifier on test set: {:.4f}'.format(recall))

    roc = roc_auc_score(lr_test_Y9, lr_y_pred)

    print('Roc of logistic regression classifier on test set: {:.4f}'.format(roc))

    print('Training data shape:{}'.format(train_X.shape))

    print('Testing data shape:{}'.format(test_X.shape))

    metrics[name] = [window_size, train_X.shape, test_X.shape, acc, precision, recall, roc]
def Metrics_NN_9(model, model_name, trainX, trainY, testX, testY, window_size):

    y_pred = model.predict(testX).round()

    score, acc = model.evaluate(testX, testY, batch_size=window_size)

    print('score: ', score)

    print('acc: ', acc)

    precision = precision_score(testY, y_pred)

    print('Precision: {:.4f}'.format(precision))

    recall = recall_score(testY, y_pred)

    print('Recall: {:.4f}'.format(recall))

    roc = roc_auc_score(testY, y_pred)

    print('ROC: {:.4f}'.format(roc))

    print('Training data shape:{}'.format(trainX.shape))

    print('Testing data shape:{}'.format(testX.shape))

    metrics[model_name] = [model.count_params(), trainX.shape, testX.shape, acc, precision, recall, roc]
def Reshape_Tensor_9(data_train, data_test, data_val, isMlp = True):

    

    if isMlp:

        data_train = np.reshape(data_train, (data_train.shape[0], data_train.shape[1] * data_train.shape[2]))

        data_test = np.reshape(data_test, (data_test.shape[0], data_test.shape[1] * data_test.shape[2])) 

        data_val = np.reshape(data_val, (data_val.shape[0], data_val.shape[1] *  data_val.shape[2]))

    else:        

        data_train = np.reshape(data_train, (data_train.shape[0], 1, data_train.shape[1] * data_train.shape[2]))

        data_test = np.reshape(data_test, (data_test.shape[0], 1, data_test.shape[1] * data_test.shape[2])) 

        data_val = np.reshape(data_val, (data_val.shape[0], 1, data_val.shape[1] *  data_val.shape[2]))

    return data_train, data_test, data_val
def fit_model_mlp_9 (X_train, y_train, window_size = 1, X_val = None, y_val = None, val = False):

    model = Sequential()

    model.add(Dense(4, bias_initializer='ones', input_dim=window_size*9, activation='tanh', name = 'Dense6'))

    model.add(Dense(1, bias_initializer = 'ones', activation='sigmoid', name = 'DenseOutput'))

    model.compile(loss = "binary_crossentropy", 

                  optimizer = "adam", metrics=["accuracy"])

    

    print(model.summary())

    

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    

    if val:

            model.fit(X_train, 

              y_train, 

              epochs = 100, 

              batch_size = window_size, 

              verbose = 2,

              validation_data=(X_val, y_val),

              callbacks = [earlystopper])

    else:

        model.fit(X_train, 

                  y_train, 

                  epochs = 100, 

                  batch_size = window_size, 

                  verbose = 1,

                  callbacks = [earlystopper])

    

    #print(model.summary())

    return(model)
def fit_model(X_train, y_train, window_size = 1, X_val = None, y_val = None, val = False):

    model = Sequential()

    

#     model.add(LSTM(1,

#                    bias_initializer='ones',

#                    input_shape = (1, window_size * 9),

#                    return_sequences=True,

#                    name='InputLayer')

#                    )

    model.add(LSTM(units=4, return_sequences=True, bias_initializer='ones', input_shape = (1, window_size * 9),

                   name = 'HiddenLayer1'))

    model.add(Dropout(0.5))

    model.add(LSTM(units=3, return_sequences=True, bias_initializer='ones', name = 'HiddenLayer2'))

    model.add(Dropout(0.4))

    model.add(LSTM(units=1, activation='softmax', bias_initializer = 'ones', name='OutputLayer'))

    model.compile(loss = "binary_crossentropy", 

                  optimizer = "adam", metrics=["accuracy"])

    print(model.summary())

    

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    if val:

            model.fit(X_train, 

              y_train, 

              epochs = 200, 

              batch_size = window_size, 

              verbose = 2,

              validation_data=(X_val, y_val),

              callbacks=[earlystopper])

    else:

        model.fit(X_train, 

                  y_train, 

                  epochs = 200, 

                  batch_size = window_size, 

                  verbose = 1,

                  callbacks=[earlystopper])

    

    return(model)
# Create test and training sets for one-step-ahead regression.

window_size_sp = 1

otrain_X9, otrain_Y9 = create_dataset(train9, window_size_sp)

otest_X9, otest_Y9 = create_dataset(test9, window_size_sp)

oval_X9, oval_Y9 = create_dataset(val9, window_size_sp)
Metrics_LR_9(otrain_X9, otrain_Y9, otest_X9, otest_Y9, "lr_1_9", window_size_sp)
mtrain_X9, mtest_X9, mval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9)
mlp1_9 = fit_model_mlp_9(mtrain_X9, otrain_Y9, window_size_sp, mval_X9, oval_Y9, True)
DrawGraphcs(mlp1_9)
Metrics_NN_9(mlp1_9, "mlp_1_9", mtrain_X9, otrain_Y9,mtest_X9, otest_Y9, window_size_sp)
ltrain_X9, ltest_X9, lval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9, False)
lstm_1_9 = fit_model(ltrain_X9, otrain_Y9, window_size_sp, lval_X9, oval_Y9, True)
DrawGraphcs(lstm_1_9)
Metrics_NN_9(lstm_1_9, "lstm_1_9", ltrain_X9, otrain_Y9, ltest_X9, otest_Y9, window_size_sp)
# Create test and training sets for one-step-ahead regression.

window_size_sp = 2

otrain_X9, otrain_Y9 = create_dataset(train9, window_size_sp)

otest_X9, otest_Y9 = create_dataset(test9, window_size_sp)

oval_X9, oval_Y9 = create_dataset(val9, window_size_sp)
Metrics_LR_9(otrain_X9, otrain_Y9, otest_X9, otest_Y9, "lr_2_9", window_size_sp)
mtrain_X9, mtest_X9, mval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9)
mlp2_9 = fit_model_mlp_9(mtrain_X9, otrain_Y9, window_size_sp, mval_X9, oval_Y9, True)
DrawGraphcs(mlp2_9)
Metrics_NN_9(mlp2_9, "mlp_2_9", mtrain_X9, otrain_Y9, mtest_X9, otest_Y9, window_size_sp)
ltrain_X9, ltest_X9, lval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9, False)
lstm_2_9 = fit_model(ltrain_X9, otrain_Y9, window_size_sp, lval_X9, oval_Y9, True)
DrawGraphcs(lstm_2_9)
Metrics_NN_9(lstm_2_9, "lstm_2_9", ltrain_X9, otrain_Y9, ltest_X9, otest_Y9, window_size_sp)
# Create test and training sets for one-step-ahead regression.

window_size_sp = 3

otrain_X9, otrain_Y9 = create_dataset(train9, window_size_sp)

otest_X9, otest_Y9 = create_dataset(test9, window_size_sp)

oval_X9, oval_Y9 = create_dataset(val9, window_size_sp)
Metrics_LR_9(otrain_X9, otrain_Y9, otest_X9, otest_Y9, "lr_3_9", window_size_sp)
mtrain_X9, mtest_X9, mval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9)
mlp3_9 = fit_model_mlp_9(mtrain_X9, otrain_Y9, window_size_sp, mval_X9, oval_Y9, True)
DrawGraphcs(mlp3_9)
Metrics_NN_9(mlp3_9, "mlp_3_9", mtrain_X9, otrain_Y9, mtest_X9, otest_Y9, window_size_sp)
ltrain_X9, ltest_X9, lval_X9 = Reshape_Tensor_9(otrain_X9, otest_X9, oval_X9, False)
lstm_3_9 = fit_model(ltrain_X9, otrain_Y9, window_size_sp, lval_X9, oval_Y9, True)
DrawGraphcs(lstm_3_9)
Metrics_NN_9(lstm_3_9, "lstm_3_9", ltrain_X9, otrain_Y9, ltest_X9, otest_Y9, window_size_sp)
def LoadHistoryFromJSON(filename):

    dic = dict()

    with open(filename, "r") as json_file:

        dic = json.load(json_file)

    return dic
model_names = ["mlp_1", "lstm_1","mlp_2", "lstm_2","mlp_3", "lstm_3", "mlp_4", "lstm_4","mlp_1_9", "lstm_1_9","mlp_2_9", "lstm_2_9","mlp_3_9", "lstm_3_9"]



histories = {}

for model_name in model_names:

    name = model_name+'_20190911_history.json'

    histories[model_name] = LoadHistoryFromJSON(name)
data = pd.DataFrame.from_dict(metrics, orient='index')

data.columns = ['nparams', 'train_shape', 'test_shape','acc','precision','recall','roc_auc']

data['name'] = data.index.values

data
# data.to_csv('Metricas.csv')
data = pd.read_csv('Metricas.csv', index_col=0)
def GraficoBarrasH(serie, title):

    bar_colors = ['yellowgreen', 'olivedrab','darkolivegreen'] * 1000

    ax = serie.plot(kind='barh', figsize=(12,6), color=bar_colors, fontsize=13, title=title)

           

    # set individual bar lables using above list

    for i in ax.patches:

        # get_width pulls left or right; get_y pushes up or down

        ax.text(i.get_width()+.005, i.get_y()+0.5, \

              str(round((i.get_width()), 3)), fontsize=10, color='black')

    

    # invert for largest on top 

    ax.invert_yaxis()
GraficoBarrasH(data['acc'],'Accuracy')
GraficoBarrasH(data['precision'], 'Precision')
GraficoBarrasH(data['recall'], 'Recall')
GraficoBarrasH(data['roc_auc'], 'Roc')
def DrawValLossFromHistories(histories, representacion):

    plt.figure(figsize=(20,8))

    legend1, legend2, legend3, legend4 = [], [], [], []



    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

    for key in histories.keys():

        if "_1" in key:

            plt.subplot(221)

            plt.plot(histories[key][representacion])

            legend1.append(key)

            plt.legend(legend1)

            plt.title('Modelos con ventana $\it{m}$=1')



        if "_2" in key:

            plt.subplot(222)

            plt.plot(histories[key][representacion])

            legend2.append(key)

            plt.legend(legend2)

            plt.title('Modelos con ventana $\it{m}$=2')



        if "_3" in key:

            plt.subplot(223)

            plt.plot(histories[key][representacion])

            legend3.append(key)

            plt.legend(legend3)

            plt.title('Modelos con ventana $\it{m}$=3')



        if "_4" in key:

            plt.subplot(224)

            plt.plot(histories[key][representacion])

            legend4.append(key)

            plt.legend(legend4)

            plt.title('Modelos con ventana $\it{m}$=4')



        plt.xlabel('epochs')

        plt.ylabel(representacion)

        plt.grid(True)
DrawValLossFromHistories(histories, "val_loss")
DrawValLossFromHistories(histories, "val_acc")
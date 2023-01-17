# Numpy e pandas

import numpy as np

import pandas as pd



# Matplotlib e Sns per plotting

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# Disabilità i warnings

import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)
house_train = pd.read_csv('../input/train.csv')

house_to_predict = pd.read_csv('../input/test.csv')



# Salviamo gli ID

ids = house_to_predict['Id']
# Funzioni di supporto già viste nel kernel EDA

def showValues (df, var):

    print (df[var].value_counts(dropna=False))

    print ("N. di categorie:",len(df[var].value_counts().tolist()))

    print ("-------------------------")

    

def showMissingData (df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    return missing_data
missing = showMissingData(house_train)

missing.head()
missing = showMissingData(house_to_predict)

missing.head()
# Feature dove si trovano i valori nan, divisi per tipo: categorico e numerico

list_cat_nan = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", 

                "GarageType", "GarageFinish", "GarageQual", "BsmtExposure", "BsmtFinType2",

                "BsmtFinType1", "BsmtCond", "BsmtQual", "MasVnrType", "Electrical", "MSZoning", 

                "Functional", "Utilities", "SaleType", "Exterior1st", "Exterior2nd", "KitchenQual"]



list_num_nan = ["LotFrontage", "GarageYrBlt", "MasVnrArea", "BsmtFullBath", "BsmtHalfBath", 

                "TotalBsmtSF", "GarageArea", "BsmtUnfSF", "GarageCars", "BsmtFinSF2", "BsmtFinSF1"]



# Usiamo queste due liste con un ciclo For per riempire i campi NaN con None (se categorici) e valore zero (se numerici)



for feature in list_cat_nan:

    house_train[feature] = house_train[feature].fillna("None")

    house_to_predict[feature] = house_to_predict[feature].fillna("None")

    

for feature in list_num_nan:

    house_train[feature] = house_train[feature].fillna(0)

    house_to_predict[feature] = house_to_predict[feature].fillna(0)

    

print ("Fatto: valori NaN risolti")
# Inizialmente uguali eccetto per SalePrice

print (house_train.shape)

print (house_to_predict.shape)
dummy_house_train = pd.get_dummies(house_train)

dummy_house_pred = pd.get_dummies(house_to_predict)



# Escluso SalePrice, adesso ci sono delle variabili categoriche che non sono presenti su pred (file di kaggle)

# Risultano quindi più variabili dummy su train rispetto a pred

print ("Train:",dummy_house_train.shape)

print ("Preds:",dummy_house_pred.shape)

dummy_house_pred.head()

dummy_house_train.head()
train_cols = dummy_house_train.columns.tolist()

preds_cols = dummy_house_pred.columns.tolist()



# Ecco le categorie mancanti

missing_cols = list(set(dummy_house_train).difference(dummy_house_pred))

print("Variabili non presenti in pred:", (missing_cols))



# SalePrice mi serve, lo rimuovo dalla lista di colonne da rimuovere

missing_cols.remove("SalePrice")



for col in missing_cols:

    dummy_house_train = dummy_house_train.drop([col], axis=1)

print ("Variabili rimosse da train")



# Adesso vanno bene

print ("Train:",dummy_house_train.shape)

print ("Preds:",dummy_house_pred.shape)
train_stats = house_train.describe().T

print("Media di SalePrice (Train): ", train_stats.loc['SalePrice']['mean'])

print("Std di SalePrice (Train): ",train_stats.loc['SalePrice']['std'])
# Normalizzazione con media e std

def norm(x, train_stats):

    df = (x - train_stats['mean']) / train_stats['std']

    return df



# Normalizzazione su ogni feature

# Fare in modo che usi mean e std del training set

def normalize(df):

    result = df.copy()

    for feature in df.columns:

        mean = df[feature].mean()

        std = df[feature].std()

        result[feature] = (df[feature] - mean) / std

    return result



# La formula inversa (ci serve per SalePrice)

def normReverse(x):

    return x * train_stats.loc['SalePrice']['std'] + train_stats.loc['SalePrice']['mean']



# Trasformazione feature in log e poi normalizzazione

norm_train = dummy_house_train.copy()

norm_train['SalePrice'] = np.log1p(house_train['SalePrice']) # va bene anche da dummy_house_train

norm_train['LotArea']   = np.log1p(house_train['LotArea'])

norm_train['1stFlrSF']  = np.log1p(house_train['1stFlrSF'])

norm_train['GrLivArea'] = np.log1p(house_train['GrLivArea'])

train_stats = norm_train.describe().T

norm_train = norm(norm_train, train_stats)



norm_pred = dummy_house_pred.copy()

norm_pred['LotArea']   = np.log1p(house_to_predict['LotArea'])

norm_pred['1stFlrSF']  = np.log1p(house_to_predict['1stFlrSF'])

norm_pred['GrLivArea'] = np.log1p(house_to_predict['GrLivArea'])

norm_pred = norm(norm_pred, train_stats)



norm_pred.head()

norm_train.head()
# Sto usando train_stats del training set e questo crea una colonna SalePrice su pred, va rimossa

# Droppiamo quelle vuote (SalePrice) e controlliamo che le colonne abbiano lo stesso ordine

norm_pred.dropna(axis='columns', inplace=True)

norm_pred = norm_pred[norm_train.drop(columns="SalePrice").columns]



print(norm_train.shape)

print(norm_pred.shape)



traincols = norm_train.columns.tolist()

predcols = norm_pred.columns.tolist()

traincols.remove("SalePrice")



# Stesso ordine se vero

print("Stesso ordine:", traincols == predcols)
# SalePrice adesso

fig, ax = plt.subplots(1,1)

fig.set_figheight(5)

fig.set_figwidth(10)

sns.distplot(norm_train['SalePrice'], bins=40, kde=True)



print("Asimmetria:",norm_train['SalePrice'].skew())

print("Curtosi:",norm_train['SalePrice'].kurtosis())
# Notare che per la riconversione vanno fatte due trasformazioni

df = normReverse(norm_train['SalePrice'])

df = np.expm1(df)
# Kaggle file

final_kaggle = norm_pred.drop(['Id'], axis=1)



# Definiamo X e Y

X = pd.DataFrame(norm_train)

X.drop(['Id', 'SalePrice'], axis=1, inplace=True)

y = norm_train['SalePrice']



# Conversione in matrici per la NN

X = X.as_matrix()

y = y.as_matrix()



# Train/Test set

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=1)



train_X.shape

train_X[0:5]



# Test_y si può riconvertire

test_y = normReverse(test_y)

test_y = np.expm1(test_y)
print (X.shape)

print (final_kaggle.shape)
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import Dense

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



def keras1_model():

    model = Sequential()

    model.add (Dense(16, input_dim=train_X.shape[1], activation=tf.nn.relu)) # train_X.shape[1] = 303 features * ogni elemento

    model.add (Dense(16, activation=tf.nn.relu))

    model.add (Dense(1, activation=tf.nn.tanh))



    model.compile(loss='mse', optimizer="adam", metrics=['mae', 'mse'])

    return model
keras1 = keras1_model()

keras1.fit (train_X, train_y, epochs=150, batch_size=train_X.shape[0]) # batch_size = tutto il dataset
normPreds = keras1.predict(train_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()



keras1_mae = mean_absolute_error(predictions, np.expm1(normReverse(train_y)))

print("Train Mean Absolute Error for Keras1: {:,.0f}".format(keras1_mae))
normPreds = keras1.predict(test_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()



keras1_mae = mean_absolute_error(predictions, test_y)

print("Validation Mean Absolute Error for Keras1: {:,.0f}".format(keras1_mae))
def build_tf_model():

    model = keras.Sequential([

    layers.Dense(32, activation=tf.nn.relu, input_shape=[train_X.shape[1]]),

    layers.Dense(32, activation=tf.nn.tanh),

    layers.Dense(1)

  ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])

    

    return model
tf_model = build_tf_model()

tf_model.summary()
# Esempio di predict su NN ancora non trained

example_batch = train_X[:10]

example_result = tf_model.predict(example_batch)

example_result = np.expm1(normReverse(example_result))

example_result
class PrintStatus(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):

        if epoch % 100 == 0: 

            print('Starting Epoch', epoch, "of", EPOCHS, "...")

        if epoch == EPOCHS-1:

            print('- Training ended')

        # print('.', end='')

        

# Training

EPOCHS = 300



history = tf_model.fit(

  train_X, train_y,

  epochs=EPOCHS, validation_split=0.2, verbose=0,

  callbacks=[PrintStatus()])
# Visualizziamo l'andamento del training

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()
# Funzione per plottare le curve di training/dev

import matplotlib.pyplot as plt



def plot_history(history, y1, y2):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error [SalePrice]')

    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')

    plt.legend()

    plt.ylim([0,y1])

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Square Error [$SalePrice$]')

    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')

    plt.legend()

    plt.ylim([0,y2])



def plot_history_noVal(history, y1, y2):

    hist = pd.DataFrame(history.history)

    hist['epoch'] = history.epoch

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error [SalePrice]')

    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')

    plt.legend()

    plt.ylim([0,y1])

    

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Square Error [$SalePrice$]')

    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')

    plt.legend()

    plt.ylim([0,y2])
# Visualizziamo le curve training/dev

plot_history(history, 1, 0.5)
# Predict su Training Set

normPreds = tf_model.predict(train_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()



mae = mean_absolute_error(predictions, np.expm1(normReverse(train_y)))

print("Train Mean Absolute Error for TFModel: {:,.0f}".format(mae))
# Predict su Test Set

normPreds = tf_model.predict(test_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()



mae = mean_absolute_error(predictions, test_y)

print("Validation Mean Absolute Error for TFModel: {:,.0f}".format(mae))
from keras import regularizers

from keras.layers import Dropout



from numpy.random import seed

from tensorflow import set_random_seed

seed(1)

set_random_seed(2)



def build_tfd_model():

    model = keras.Sequential([

    layers.Dense(120, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.27), input_shape=[train_X.shape[1]]),

    layers.Dense(120, activation=tf.nn.tanh, kernel_regularizer=regularizers.l2(0.27)),

    layers.Dense(1)

  ])

    learning_rate_adam = 0.0001

    optimizer = tf.keras.optimizers.Adam(learning_rate_adam)

    # learning_rate_rmsprop = 0.0003

    # learning_rate_SGD = 0.01

    # optimizer = tf.keras.optimizers.SGD(learning_rate_SGD)

    # optimizer = tf.keras.optimizers.RMSprop(learning_rate_rmsprop)

    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    

    return model



tfd_model = build_tfd_model()

tfd_model.summary()
# Con Dev Validation

tfd_model = build_tfd_model()

EPOCHS = 2000



history = tfd_model.fit(

  X, y, batch_size=X.shape[0],  # X.shape[0] = tutto il dataset

  epochs=EPOCHS, validation_split=0.20, verbose=0,

  callbacks=[PrintStatus()])



# Visualizziamo l'andamento del training

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()



plot_history(history, 1, 1)
hist.tail()

# val_mean_abs_error: 0.2123-0.2147
# Senza Dev Validation

tfd_model = build_tfd_model()

EPOCHS = 2000



history = tfd_model.fit(

  train_X, train_y, batch_size=train_X.shape[0],  # train_X.shape[0] = tutto il train set

  epochs=EPOCHS, validation_split=0, verbose=0,   # nessun validation split

  callbacks=[PrintStatus()])



# Visualizziamo l'andamento del training

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch



plot_history_noVal(history, 1, 1)
hist.tail()
# Predict su Training Set

normPreds = tfd_model.predict(train_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()

mae = mean_absolute_error(predictions, np.expm1(normReverse(train_y))) # np.expm1(train_y)

print("Training Mean Absolute Error for TFDModel: {:,.0f}".format(mae))



# Predict su Test Set

normPreds = tfd_model.predict(test_X)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()



mae = mean_absolute_error(predictions, test_y)

print("Validation Mean Absolute Error for TFDModel: {:,.0f}".format(mae))
tfd_model = build_tfd_model()

EPOCHS = 2000



history = tfd_model.fit(

  X, y, batch_size=X.shape[0], # tutto il dataset

  epochs=EPOCHS, validation_split=0, verbose=0,

  callbacks=[PrintStatus()])
# Predict su final kaggle

normPreds = tfd_model.predict(final_kaggle)

predictions = np.expm1(normReverse(normPreds))

predictions = predictions.squeeze()

predictions.shape
sub = pd.DataFrame()

sub['Id'] = ids

sub['SalePrice'] = predictions

sub.to_csv('submission.csv', index=False)

sub
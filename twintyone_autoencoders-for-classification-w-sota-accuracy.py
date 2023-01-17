import numpy as np

import pandas as pd



from sklearn import metrics

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation



import warnings

warnings.simplefilter('ignore')
data = pd.read_csv("../input/gene-expression-cancer-rnaseq/data.csv")

labels = pd.read_csv("../input/gene-expression-cancer-rnaseq/labels.csv")
data
labels
data = data.drop(columns=['Unnamed: 0'])

labels = labels.drop(columns=['Unnamed: 0'])
print(data.shape)

print(labels.shape)
labels.Class.value_counts()
data.isnull().values.any()
(data.min() == data.max()).value_counts()
(data.max() == 0.0).value_counts()
data = data.loc[:, (data.max() != data.min())]

data.shape
df, df_test = train_test_split(data, test_size=0.25, random_state=21)



test_index = list(df_test.index)

df_test = df_test.values
brca_mask = labels['Class'] == 'BRCA'

kirc_mask = labels['Class'] == 'KIRC'

luad_mask = labels['Class'] == 'LUAD'

prad_mask = labels['Class'] == 'PRAD'

coad_mask = labels['Class'] == 'COAD'



df_brca = df[brca_mask]

df_kirc = df[kirc_mask]

df_luad = df[luad_mask]

df_prad = df[prad_mask]

df_coad = df[coad_mask]



x_brca = df_brca.values

x_kirc = df_kirc.values

x_luad = df_luad.values

x_prad = df_prad.values

x_coad = df_coad.values
def Model():

    model = Sequential()

    model.add(Dense(250, input_dim=x_brca.shape[1], activation='relu'))

    model.add(Dense(21, activation='relu'))

    model.add(Dense(250, activation='relu'))

    model.add(Dense(x_brca.shape[1]))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss',

                               min_delta = 0,

                               patience = 5,

                               verbose = 1,

                               restore_best_weights = True)
def evaluate(model):

    return (np.sqrt(metrics.mean_squared_error(model.predict(x_brca), x_brca)),

            np.sqrt(metrics.mean_squared_error(model.predict(x_kirc), x_kirc)),

            np.sqrt(metrics.mean_squared_error(model.predict(x_luad), x_luad)),

            np.sqrt(metrics.mean_squared_error(model.predict(x_prad), x_prad)),

            np.sqrt(metrics.mean_squared_error(model.predict(x_coad), x_coad)))
x_brca_train, x_brca_test = train_test_split(x_brca, test_size=0.25, random_state=21)



brca_model = Model()

brca_model.fit(x_brca_train, x_brca_train, validation_data=(x_brca_test, x_brca_test), verbose=1, epochs=100, callbacks=[early_stopping])

evaluate(brca_model)
x_kirc_train, x_kirc_test = train_test_split(x_kirc, test_size=0.25, random_state=21)



kirc_model = Model()

kirc_model.fit(x_kirc_train, x_kirc_train, validation_data=(x_kirc_test, x_kirc_test), verbose=1, epochs=100, callbacks=[early_stopping])

evaluate(kirc_model)
x_luad_train, x_luad_test = train_test_split(x_luad, test_size=0.25, random_state=21)



luad_model = Model()

luad_model.fit(x_luad_train, x_luad_train, validation_data=(x_luad_test, x_luad_test), verbose=1, epochs=100, callbacks=[early_stopping])

evaluate(luad_model)
x_prad_train, x_prad_test = train_test_split(x_prad, test_size=0.25, random_state=21)



prad_model = Model()

prad_model.fit(x_prad_train, x_prad_train, validation_data=(x_prad_test, x_prad_test), verbose=1, epochs=100, callbacks=[early_stopping])

evaluate(prad_model)
x_coad_train, x_coad_test = train_test_split(x_coad, test_size=0.25, random_state=21)



coad_model = Model()

coad_model.fit(x_coad_train, x_coad_train, validation_data=(x_coad_test, x_coad_test), verbose=1, epochs=100, callbacks=[early_stopping])

evaluate(coad_model)
models = [brca_model, kirc_model, luad_model, prad_model, coad_model]
def get_pred(df_test, models):

    pred_class = []

    for i in range(len(df_test)):

        loss = []

        x = df_test[i].reshape(1, 20264)

        for model in models:

            loss.append(np.sqrt(metrics.mean_squared_error(model.predict(x), x)))

        pred_class.append(loss.index(min(loss)))

    return pred_class
def get_label(test_index):

    num_label = []

    for l in range(len(test_index)):

        _ = labels.values[test_index[l]][0][0]

        if _ == 'B': num_label.append(0)

        elif _ == 'K': num_label.append(1)

        elif _ == 'L': num_label.append(2)

        elif _ == 'P': num_label.append(3)

        elif _ == 'C': num_label.append(4)

    return num_label
pred_correct = 0

num_label = get_label(test_index)

pred_class = get_pred(df_test, models)



for i in range(len(num_label)):

    if num_label[i] == pred_class[i]: pred_correct += 1
print('Accuracy:', pred_correct/len(num_label) * 100)
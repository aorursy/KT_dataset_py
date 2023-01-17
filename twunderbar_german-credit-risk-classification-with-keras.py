import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def normalize(df):

    result = df.copy()

    max_value = df.max()

    min_value = df.min()

    result = (df - min_value) / (max_value - min_value)

    return result



from pandas.api.types import is_string_dtype



data = pd.read_csv('../input/german-credit-data-with-risk/german_credit_data.csv',index_col=0,sep=',')

labels = data.columns

# lets go through column 2 column

for col in labels:

    if is_string_dtype(data[col]):

        if col == 'Risk':

            # we want 'Risk' to be a binary variable

            data[col] = pd.factorize(data[col])[0]

            continue

        # the other categorical columns should be one-hot encoded

        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)

        data.drop(col, axis=1, inplace=True)

    else:

        data[col] = normalize(data[col])



# move 'Risk' back to the end of the df

data = data[[c for c in data if c not in ['Risk']] + ['Risk']]



data_train = data.iloc[:800]

data_valid = data.iloc[800:]

x_train = data_train.iloc[:,:-1]

y_train = data_train.iloc[:,-1]

x_val = data_valid.iloc[:,:-1]

y_val = data_valid.iloc[:,-1]
from keras.models import Sequential

from keras import regularizers

from keras import optimizers

from keras.layers import Dense, Dropout



sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)



model = Sequential()

model.add(Dense(units=50, activation='tanh', input_dim=24, kernel_initializer='glorot_normal', bias_initializer='zeros'))#, kernel_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.35))

model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))

model.compile(loss='binary_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])



model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values), epochs=30, batch_size=128)
model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values), epochs=30, batch_size=128)
y_pred = model.predict_classes(x_val.values)

y_val = y_val.values



from sklearn.metrics import confusion_matrix, precision_score

import seaborn as sns



sns.heatmap(confusion_matrix(y_val,y_pred),annot=True,fmt='.5g') 

print('Precision Score on validation data is {}'.format(precision_score(y_val, y_pred, average='weighted')))
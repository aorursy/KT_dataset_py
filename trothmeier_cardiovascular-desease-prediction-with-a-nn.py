# Imports

import numpy as np 

import pandas as pd 

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold

from sklearn.preprocessing import StandardScaler

from keras.optimizers import *

from keras.initializers import *

from keras.models import *

from keras.layers import *

from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score



# Get data

df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', delimiter=';')

df.drop('id', axis=1, inplace=True)

df.head()
df.describe()
df[df['ap_lo'] >= df['ap_hi']]
df.drop(df[df["ap_lo"] > df["ap_hi"]].index, inplace=True)

df.drop(df[df["ap_lo"] <= 30].index, inplace=True)

df.drop(df[df["ap_hi"] <= 40].index, inplace=True)

df.drop(df[df["ap_lo"] >= 200].index, inplace=True)

df.drop(df[df["ap_hi"] >= 250].index, inplace=True)

df[['ap_lo', 'ap_hi']].describe()
X = df.drop('cardio', axis=1)

Y = df['cardio']



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

s = StandardScaler()

x_train = s.fit_transform(x_train)

x_test = s.transform(x_test)



# Split train set in train and validation set:

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=0)
# Silence warnings

import warnings as w

w.simplefilter('ignore')





def create_model():

    # Hyperparameter:

    init_w = glorot_uniform(seed=0)

    loss = "binary_crossentropy"

    optimizer = Adadelta()

    

    # Defining the model:

    model = Sequential()



    model.add(Dense(50, kernel_initializer=init_w, input_shape=(x_train.shape[1],)))

    model.add(BatchNormalization())

    model.add(LeakyReLU())

    model.add(Dropout(rate=0.1))



    model.add(Dense(25, kernel_initializer=init_w))

    model.add(BatchNormalization())

    model.add(LeakyReLU())

    model.add(Dropout(rate=0.1))



    model.add(Dense(12, kernel_initializer=init_w))

    model.add(LeakyReLU())



    model.add(Dense(1, kernel_initializer=init_w))

    model.add(Activation("sigmoid"))

    

    model.summary()

    

    # Training

    model.compile(

        loss=loss,

        optimizer=optimizer,

        metrics=["accuracy"])



    return model
nn = create_model()

nn.fit(

    x=x_train,

    y=y_train,

    verbose=2,

    epochs=50,

    batch_size=256,

    validation_data=[x_valid, y_valid])
# Testing

test_score = nn.evaluate(x_test, y_test)

print("Testing Acc:", test_score[1])
y_pred = nn.predict(x_test)

cm = confusion_matrix(y_test, y_pred.round())

print("Confusion Matrix:", "\n", cm)
tpr, fpr, threshold = roc_curve(y_test, y_pred)

auc_score = roc_auc_score(y_test, y_pred)

print("AUC-score:", auc_score)
%matplotlib inline

import matplotlib.pyplot as plt



plt.plot(tpr, fpr)

plt.show()
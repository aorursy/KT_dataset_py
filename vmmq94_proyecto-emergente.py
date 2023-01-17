import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns



from keras.layers import Input, Dense, BatchNormalization, Add, GaussianNoise, Dropout

from keras.models import Model

from sklearn.metrics import roc_auc_score

from keras.layers import Wrapper

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from keras import regularizers

# Feature Scaling

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))





import warnings

warnings.filterwarnings("ignore")



sns.set() 
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

print ("Train: ",train_df.shape)

print ("Test: ",test_df.shape)
train_df.head()
test_df.head()
print(train_df.isnull().sum())

print(test_df.isnull().sum())
train_test = [train_df, test_df]
for data in train_test:

    data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)

    data['Age'] = data['Age'].astype(int)

    

    data["Embarked"].fillna(data['Embarked'].value_counts().idxmax(), inplace=True)

    data['Embarked'] = data['Embarked'].map( {'S': 2, 'C': 1, 'Q': 0} ).astype(int)

    

    data["Fare"].fillna(data["Fare"].median(skipna=True), inplace=True)

    data['Fare'] = data['Fare'].astype(int)

    

    data.drop('Cabin', axis=1, inplace=True)

    

    data.drop('Ticket', axis=1, inplace=True)

    

   

    

    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        

    data['Title'] = data["Name"].str.extract(' ([A-Za-z]+)\.')

    data.drop('Name', axis=1, inplace=True)

    data.drop('Title', axis=1, inplace=True)

print(train_df.isnull().sum())

print(test_df.isnull().sum())
train_df.head()
sns.barplot(x='Pclass', y='Survived', data=train_df)
sns.barplot(x='Embarked', y='Survived', data=train_df)
sns.barplot(x='Sex', y='Survived', data=train_df)
sns.barplot(x='Survived', y='Fare', data=train_df)
train_df = train_df.drop(['PassengerId'], axis=1)
#X_train = train_df.drop('Survived', axis=1)

#y_train = train_df['Survived']

#X_test = test_df.drop("PassengerId", axis=1).copy()



#X_train.shape, y_train.shape, X_test.shape
train_dfX = train_df.drop('Survived', axis=1)

train_dfY = train_df['Survived']

submission = test_df[['PassengerId']].copy()

test_df = test_df.drop("PassengerId", axis=1).copy()

precisiones_globales=[]

epochs = 26

def graf_model(train_history):

    f = plt.figure(figsize=(15,10))

    ax = f.add_subplot(121)

    ax2 = f.add_subplot(122)

    # summarize history for accuracy

    ax.plot(train_history.history['binary_accuracy'])

    ax.plot(train_history.history['val_binary_accuracy'])

    ax.set_title('model accuracy')

    ax.set_ylabel('accuracy')

    ax.set_xlabel('epoch')

    ax.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    ax2.plot(train_history.history['loss'])

    ax2.plot(train_history.history['val_loss'])

    ax2.set_title('model loss')

    ax2.set_ylabel('loss')

    ax2.set_xlabel('epoch')

    ax2.legend(['train', 'test'], loc='upper left')

    plt.show()

def precision(model, registrar=False):

    y_pred = model.predict(train_dfX)

    train_auc = roc_auc_score(train_dfY, y_pred)

    y_pred = model.predict(val_dfX)

    val_auc = roc_auc_score(val_dfY, y_pred)

    print('Train AUC: ', train_auc)

    print('Vali AUC: ', val_auc)

    if registrar:

        precisiones_globales.append([train_auc,val_auc])
sc = StandardScaler()

train_dfX = sc.fit_transform(train_dfX)

test_df = sc.transform(test_df)
train_dfX,val_dfX,train_dfY, val_dfY = train_test_split(train_dfX,train_dfY , test_size=0.2, stratify=train_dfY)

print("Entrnamiento: ",train_dfX.shape)

print("Validacion : ",val_dfX.shape)
def func_model_reg():   

    inp = Input(shape=(7,))

    x=Dropout(0)(inp)

    x=Dense(4096, activation="relu",  bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0)(x)

    x=Dense(4096, activation="relu",  bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0)(x)

    x=Dense(2048, activation="relu",  bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0)(x)

    x=Dense(1024, activation="relu",  bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01))(x)

    x=Dropout(0)(x)

    x=Dense(1024, activation="relu",  bias_initializer='zeros', kernel_regularizer=None)(x)

    x=Dropout(0)(x)  

    x=Dense(1, activation="sigmoid", bias_initializer='zeros')(x) 

    model = Model(inputs=inp, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

    return model
model1 = func_model_reg()

print(model1.summary())

train_history_tam1 = model1.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY), verbose=0)

graf_model(train_history_tam1)

precision(model1)
modelRF = func_model_reg()

print(modelRF.summary())

train_history_regF = modelRF.fit(train_dfX, train_dfY, batch_size=512, epochs=epochs, validation_data=(val_dfX, val_dfY))

graf_model(train_history_regF)

precision(modelRF, True)

y_test = modelRF.predict(test_df)

submission['Survived'] = np.round(y_test).astype(int)

submission.to_csv('submission.csv', index=False)

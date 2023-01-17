import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from sklearn.preprocessing import LabelEncoder, StandardScaler



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout



from glob import glob



from keras.callbacks import ModelCheckpoint
train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')
def plot(history, arr):

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    for idx in range(2):

        ax[idx].grid(True)

        ax[idx].plot(history.history[arr[idx][0]], lw = 3)

        ax[idx].plot(history.history[arr[idx][1]],  dashes=[6, 2], lw = 3)

        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)

        ax[idx].set_xlabel('Epoch ',fontsize=16)

        ax[idx].set_ylabel('Metric',fontsize=16)

        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)

        

        

def cleanner(data):

    le = LabelEncoder()

    

    data.Sex.replace('male',   0, inplace = True)

    data.Sex.replace('female', 1, inplace = True)



    data.Embarked.fillna(      0, inplace = True)

    data.Embarked.replace('S', 0, inplace = True)

    data.Embarked.replace('C', 1, inplace = True)

    data.Embarked.replace('Q', 2, inplace = True)

    

    

    data.Age.fillna(int(data.Age.mean()), inplace = True)

    data.Fare.fillna(data.Fare.mean(), inplace = True)

    data['Ticket'] = le.fit_transform(data['Ticket'])

    

    data.Cabin.fillna('0', inplace = True)

    for x in data.Cabin:

        data.Cabin.replace(x, x[0], inplace = True)



    for index, x in enumerate(list(data.Cabin.str[0].unique())):

        data.Cabin.replace(x, index, inplace = True)
cleanner(train); 

cleanner(test)
plt.figure(figsize=(12,10))

cor = train.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds);



cor_target = abs(cor.Survived)

cor_target[cor_target > 0.1]
X_train_SUB = train._get_numeric_data()

TARGET = X_train_SUB['Survived']

X_train_SUB = X_train_SUB.drop(columns = 'Survived')

X_test_SUB  = test._get_numeric_data()
selects = ['Pclass', 'Sex', 'Fare', 'Cabin','Ticket', 'Embarked']



scaler = StandardScaler()

TRAIN  = scaler.fit_transform(X_train_SUB)

SUB    = scaler.transform(X_test_SUB)

X_train, X_test, y_train, y_test = train_test_split(TRAIN, TARGET, test_size = 0.3, random_state = 0)
batch_size = 8

epochs = 100



model = Sequential()

model.add(Dense(16, activation = "relu", input_dim = X_train.shape[1]))

model.add(Dropout(0.5))

model.add(Dense(8, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "rmsprop", loss= "binary_crossentropy", metrics = ["accuracy"])



checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_accuracy',

    mode='auto',

    verbose=1,

    save_best_only=True

)



History = model.fit(X_train, 

                    y_train, 

                    epochs=epochs, 

                    batch_size=batch_size, 

                    validation_data = [X_test, y_test],

                    callbacks = [checkpoint])
loss, accu = model.evaluate(X_test, y_test)

print("Acurracy: ", accu)

print("Losses", loss)
plot(History, [['loss', 'val_loss'], ['accuracy', 'val_accuracy']])
model.load_weights('model.h5')



preds = model.predict_classes(X_test)



print('Accuracy Score : '  + str(accuracy_score( y_test, preds)))

print('Precision Score : ' + str(precision_score(y_test, preds)))

print('Recall Score : '    + str(recall_score(   y_test, preds)))

print('F1 Score : '        + str(f1_score(       y_test, preds)))

print("=" * 35)
preds = model.predict_classes(SUB).reshape(-1)

sub = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds})

sub.to_csv('submission.csv', index = False)
sub
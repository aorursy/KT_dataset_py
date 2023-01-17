import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/titanic/train.csv')
X_test_file = pd.read_csv('../input/titanic/test.csv')
train.head()
def impute_age(cols):
    Age=cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

#APPLY THIS FUNCTION ON AGE COLUMN
train['Age']= train[['Age', 'Pclass']].apply(impute_age, axis=1)
#CREATE HEATMAP TO OBSERVE CHANGES
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#NOW WE WILL DROP THE CABIN COLUMN SINCE ALOT OF VALUES ARE MISSING
train.drop('Cabin', axis=1,inplace=True)
#AGAIN CHECK THE HEATMAP FOR NULL VALUES
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#JUST IN CASE PERFORM A DROPNA TO BE SURE
train.dropna(inplace=True)

#DUMMY DATA!!!!
#CReATING DUMMY DATA FOR CATEGORICAL FIELDS
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
#CONCAT THESE FIELDS IN THE DATA
train = pd.concat([train, sex, embark], axis=1)

#DROP COULMNS WHICH YOU DONT NEED
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace= True)

train.head()
X_test_file['Fare'].fillna(0, inplace=True)
X_test_file[X_test_file['Fare'].isnull()]
#APPLY THIS FUNCTION ON AGE COLUMN
X_test_file['Age']= X_test_file[['Age', 'Pclass']].apply(impute_age, axis=1)
#CREATE HEATMAP TO OBSERVE CHANGES
sns.heatmap(X_test_file.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#NOW WE WILL DROP THE CABIN COLUMN SINCE ALOT OF VALUES ARE MISSING
X_test_file.drop('Cabin', axis=1,inplace=True)
#AGAIN CHECK THE HEATMAP FOR NULL VALUES
sns.heatmap(X_test_file.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#JUST IN CASE PERFORM A DROPNA TO BE SURE
X_test_file.dropna(inplace=True)

#DUMMY DATA!!!!
#CReATING DUMMY DATA FOR CATEGORICAL FIELDS
sex_1 = pd.get_dummies(X_test_file['Sex'], drop_first = True)
embark_1 = pd.get_dummies(X_test_file['Embarked'], drop_first = True)
#CONCAT THESE FIELDS IN THE DATA
X_test_file = pd.concat([X_test_file, sex_1, embark_1], axis=1)

#DROP COULMNS WHICH YOU DONT NEED
X_test_file.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace= True)

X = train.drop('Survived', axis=1)
#X = X[['Fare', 'male']]
y = train['Survived']
from sklearn.feature_selection import SelectKBest,chi2
test=SelectKBest(score_func=chi2,k=2)
fit=test.fit(X,y)
print(fit.scores_)
X.head()
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_validate)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 4, activation = 'relu'))
#model.add(Dropout(0.5))

#model.add(Dense(units = 6, activation = 'relu'))
#model.add(Dense(units = 4, activation = 'relu'))
#model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))



model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', mode= 'min', verbose= 1, patience=15)

model.fit(x=X_train, y=y_train, epochs = 2000, validation_data=(X_validate, y_validate), batch_size=128, callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
losses.plot()
X_test_file = scaler.transform(X_test_file)
#model.predict_classes(X_test_file)
p = model.predict_classes(X_validate)
from sklearn.metrics import classification_report
print(classification_report(y_validate, p))
163/267
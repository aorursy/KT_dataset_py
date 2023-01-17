%matplotlib inline



# Import Pandas for Data Loading & Manipulation

import pandas as pd



# Data Visualization Imports

import seaborn as sns

import matplotlib as plt



# Preprocessing Data

from sklearn.preprocessing import LabelEncoder



# Keras: Deep Learning Library

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense
data_train = pd.read_csv('../input/train.csv')

data_train.set_index(['PassengerId'],inplace=True)
data_test = pd.read_csv('../input/test.csv')

data_test.set_index(['PassengerId'], inplace=True)
data_train.head()
data_train.describe()
data_train.info()
data_train.groupby('Sex').Survived.mean().plot(kind='bar')
data_train.groupby('Sex').Survived.mean().plot(kind='bar')
data_train.groupby('SibSp').Survived.mean().plot(kind='bar')
data_train.groupby('Parch').Survived.mean().plot(kind='bar')
sns.factorplot("Sex", "Survived",hue='Pclass', data=data_train)
le=LabelEncoder()

le.fit(data_train['Sex'])

print("The values of Sex are: "+str(le.classes_))

print("The mapped values of Sex as intergers are: "+str(le.transform(le.classes_)))

data_train['Sex'] = le.fit_transform(data_train['Sex'])

data_train['title'] = data_train.apply(lambda x: x['Name'].split(',')[1].split('.')[0], axis=1)

le.fit(data_train['title'])

data_train['title'] = le.fit_transform(data_train['title'])



    

data_train['familysize'] = data_train['SibSp']+data_train['Parch']

data_train['isalone'] = 0

data_train['isalone'] = data_train.loc[data_train['familysize'] == 1, 'isalone'] = 1

data_train['Embarked'] = data_train['Embarked'].fillna('C')

data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)
data_imp_var_train = data_train[['Pclass', 'Sex', 'isalone', 'familysize', 'SibSp', 'Parch', 'Embarked']].as_matrix()

data_survived = to_categorical(data_train['Survived'])
model = Sequential()

model.add(Dense(200, activation='relu', input_shape=(data_imp_var_train.shape[1],)))

model.add(Dense(200, activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data_imp_var_train, data_survived, epochs=200)
data_test['familysize'] = data_test['SibSp']+data_test['Parch']

data_test['isalone'] = 0

data_test['isalone'] = data_test.loc[data_test['familysize'] == 1, 'isalone'] = 1

data_test['Embarked'] = data_test['Embarked'].fillna('C')

data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

data_test['title'] = data_test.apply(lambda x: x['Name'].split(',')[1].split('.')[0], axis=1)



data_test_wanted = data_test[['Pclass', 'Sex', 'isalone', 'familysize', 'SibSp', 'Parch', 'Embarked']]

le.fit(data_test_wanted['Sex'])

data_test_wanted['Sex'] = le.fit_transform(data_test_wanted['Sex'])



# le.fit(data_test_wanted['title'])

# data_test_wanted['title'] = le.fit_transform(data_test_wanted['title'])

# data_test_wanted.head()
data_test['Survived']=model.predict_classes(data_test_wanted.as_matrix())

data_test[['Survived']].to_csv('output_fs_dp.csv', index=True)
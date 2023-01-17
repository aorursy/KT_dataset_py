import pandas as pd

import numpy as np

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import np_utils  # for transforming data later



from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



np.random.seed(1)  # for reproducability
train = pd.read_csv( '../input/train.csv')

test = pd.read_csv('../input/test.csv')



train.head(10)
full_data = [train, test]



# Feature that tells whether a passenger had a cabin on the Titanic

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# some of the feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column (use most common value 'S')

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and fill with median

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Remove all NULLS in the Age column and fill random age in range[avg - std, avg + std] (questionable...)

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()  # counts the number of NaN is the Age column

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    # fill missing ages with random ages around mean age

    dataset.loc[dataset['Age'].isnull(), 'Age'] = age_null_random_list  

    dataset['Age'] = dataset['Age'].astype(int)

# Map Sex to (ordered) categorical [1,0]

for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].astype("category").cat.codes



# create dummies for Embarked (unordered categorical feature)

train = pd.get_dummies(train, columns=['Embarked'])

test = pd.get_dummies(test, columns=['Embarked'])

    
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']

train = train.drop(drop_elements, axis = 1)

test = test.drop(drop_elements, axis = 1)



train.head(10)
# this feature set gives reasonable accuracy

wanted_features = ['Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize']



X_train = train[wanted_features].values

X_test = test[wanted_features].values



Y_train = train['Survived'].values



n_features = X_train.shape[1]
scaler = StandardScaler()  

scaler.fit(X_train)  



X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test)
optim = 'rmsprop'



dim1 = 40

drop1 = 0.5

act1 = 'relu'



dim2 = 20

drop2 = 0.5

act2 = 'relu'



model = Sequential()

# input -> first layer

model.add(Dense(dim1, input_dim=n_features, activation=act1))

model.add(Dropout(drop1))



# first -> second layer

model.add(Dense(dim2, activation=act2))

model.add(Dropout(drop2))



# second -> output layer

model.add(Dense(1, activation='sigmoid'))

print(model.summary())



model.compile(loss='binary_crossentropy',

              optimizer=optim,

              metrics=['accuracy'])
epochs = 30

batch_size = 20



# use fraction of training data as validation data to infer model generalizability

val_split = 0.2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=val_split)



model.fit(X_train, Y_train,

          validation_data=(X_val, Y_val),

          epochs=epochs,

          batch_size=batch_size)
Y_pred = model.predict(X_val)

surv_msk = Y_pred > 0.5

Y_pred = np.zeros(Y_pred.shape)

Y_pred[surv_msk] = 1.



n_train = len(Y_train)

n_test = len(Y_val)

train_frac = Y_train.sum()/n_train

pred_frac = Y_pred.sum()/n_train

test_frac = Y_val.sum()/n_train



print('validation set contains %i samples' % n_test)



print('%2.2f%% survived in training set' % (100*train_frac))

print('%2.2f%% survived in validation set' % (100*test_frac))

print('%2.2f%% survived in predicted test set' % (100*pred_frac))



acc = accuracy_score(Y_val, Y_pred)

print('Accuracy on validation set: %2.2f%%' % (100*acc))
cm = confusion_matrix(Y_val, Y_pred)

f_score = f1_score(Y_val, Y_pred)

print('Confusion Matrix:')

print(cm)



cm_percent = 100*cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

print('\nin percent per row:')

print(cm_percent)



print('\n')

print('%02.2f%% of survived predicted correctly (true positive)' % cm_percent[1,1])

print('%02.2f%% of survived predicted INcorrectly (false positive)' % cm_percent[1,0])



print('\n')

print('%02.2f%% of deceased predicted correctly (true negative)' % cm_percent[0,0])

print('%02.2f%% of deceased predicted INcorrectly (false negative)' % cm_percent[0,1])



print('\nF-Score: %2.4f' % (f_score))
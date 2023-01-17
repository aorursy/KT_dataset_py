import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
train_data = pd.read_csv('../input/titanic/train.csv')
train_data.head(10)
train_data.info()
train_data.describe(include = "all")
train_data.isna().sum()
train_data.drop("PassengerId", axis=1, inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop("Name", axis=1, inplace=True)
train_data.drop("Ticket", axis=1, inplace=True)
train_data['Age'] = train_data['Age'].fillna(train_data.groupby('Sex')['Age'].transform('mean'))
train_data.isna().sum()
# Embarked feature
print(train_data["Embarked"].unique())
print(np.sum([train_data["Embarked"]=="S"]))
print(np.sum([train_data["Embarked"]=="C"]))
print(np.sum([train_data["Embarked"]=="Q"]))
train_data.fillna({"Embarked":"S"},inplace=True)
#Map the categorical colum 'Sex' with numerical data
sex_mapping = {"male": 0, "female": 1}
train_data['Sex'] = train_data['Sex'].map(sex_mapping)
train_data.head()
#Mapping for Embarked feature
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
train_data.head()
X= train_data.drop(["Survived"],axis=1)
Y= train_data["Survived"]
X.shape,Y.shape
import tensorflow as tf
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# construct model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(7,), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('int32').reshape(-1,1)

#
print('\n Training .... ')
hist = model.fit(x_train, y_train,
                 batch_size=64,
                 epochs=200,
                 verbose=0,
                 validation_split=0.1)

print('\n accuracy after last training epoch | {0} |'.format(hist.history['accuracy'][-1]))
print('\n Evaluation .... ')
results = model.evaluate(x_test, y_test, verbose=0)
print('\n accuracy of evaluation on test data non seen before | {0} |'.format(results[1]))

test_data = pd.read_csv('../input/titanic/test.csv')
test_data.drop('Cabin', axis = 1, inplace=True)
test_data.drop("Name",axis=1,inplace=True)
test_data.drop("Ticket",axis=1,inplace=True)
# #
test_data['Age'] = test_data['Age'].fillna(train_data.groupby('Sex')['Age'].transform('mean'))
test_data.fillna({"Embarked":"S"},inplace=True)
test_data['Sex'] = test_data['Sex'].map(sex_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)
test_data['Fare'].fillna(np.mean(test_data['Fare']), inplace=True)
ids = test_data['PassengerId']
predictions = model.predict(test_data.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
predictions = predictions.reshape((predictions.shape[0],))
predictions = np.where(predictions >= 0.5, 1, 0)
output = pd.DataFrame({ 'PassengerId' : np.array(ids), 'Survived': np.array(predictions) })
output.to_csv('submission.csv', index=False)
print('done')

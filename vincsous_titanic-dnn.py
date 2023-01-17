# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
raw_data_training = pd.read_csv('/kaggle/input/titanic/train.csv')

df_train = raw_data_training.copy()

df_train
raw_data_testing = pd.read_csv('/kaggle/input/titanic/test.csv')

df_test = raw_data_testing.copy()

df_test
df_train.info() # get data set info to see what to keep/drop
df_train = df_train.drop(['PassengerId'], axis = 1) # PassengerId will not provide relevant information

df_test = df_test.drop(['PassengerId'], axis = 1) 



df_train = df_train.drop(['Cabin'], axis = 1) # Cabin has too many missing data. About 75%

df_test = df_test.drop(['Cabin'], axis = 1)



df_train = df_train.drop(['Ticket'], axis = 1) # Too many tickets name, and probably highly correlated with Pclass and Embarked

df_test = df_test.drop(['Ticket'], axis = 1)



df_train
y_train = df_train['Survived']

x_train = df_train.drop(['Survived'], axis = 1)



x_test = df_test
# For the training dataset:

x_train['Fam'] = x_train['SibSp'] + x_train['Parch']

x_train = x_train.drop(['SibSp', 'Parch'], axis = 1)



x_train.head()
# For the testing dataset:

x_test['Fam'] = x_test['SibSp'] + x_test['Parch']

x_test = x_test.drop(['SibSp', 'Parch'], axis = 1)



x_test.head()
## Sex: replace male by 0 and female by 1



x_train['Sex'] = x_train['Sex'].map({'male':0, 'female':1})

x_test['Sex'] = x_test['Sex'].map({'male':0, 'female':1})
# Find out how many titles are there:



title = [i.split(",")[1].split(".")[0].strip() for i in x_train["Name"]]



title = pd.DataFrame(title)

title[0].unique()

# Do we have the same in the test dataset



title_test = [i.split(",")[1].split(".")[0].strip() for i in x_test["Name"]] # I took this from someones Kernel



title_test = pd.DataFrame(title_test)

title_test[0].unique()
title_test[0].unique() == title[0].unique()
# I replace 'Miss', 'Mr', 'Mrs', 'Mlle', 'Ms', 'Mme' with 0

# I replace 'Master', 'Jonkheer', 'Col', 'Sir', 'Major', 'Dr', 'Lady', 'the Countess', 'Rev', 'Don','Capt','Dona' with 1



title[0] = title[0].map({'Miss':0, 'Mr':0, 'Mrs':0, 'Mlle':0, 'Ms':0, 'Mme':0,

                  'Master':1, 'Jonkheer':1, 'Col':1, 'Sir':1, 'Major':1, 'Dr':1, 'Lady':1, 'the Countess':1, 'Rev':1, 'Don':1,'Capt':1})



title_test[0] = title_test[0].map({'Miss':0, 'Mr':0, 'Mrs':0, 'Ms':0,

                  'Master':1, 'Col':1, 'Dr':1, 'Rev':1, 'Dona':1})
x_train['Title'] = title[0]

x_train = x_train.drop(['Name'], axis = 1)



x_test['Title'] = title_test[0]

x_test = x_test.drop(['Name'], axis = 1)
# Training dataset

Pclass_dummies_train = pd.get_dummies(x_train['Pclass'])

x_train_no_Pclass = x_train.drop(['Pclass'], axis = 1)

x_train = pd.concat([x_train_no_Pclass, Pclass_dummies_train], axis=1)
# Testing dataset

Pclass_dummies_test = pd.get_dummies(x_test['Pclass'])

x_test_no_Pclass = x_test.drop(['Pclass'], axis = 1)

x_test = pd.concat([x_test_no_Pclass, Pclass_dummies_test], axis=1)
Fam_dummies_train = pd.get_dummies(x_train['Fam'])

x_train_no_Fam = x_train.drop(['Fam'], axis = 1)

x_train = pd.concat([x_train_no_Fam, Fam_dummies_train], axis=1)
Fam_dummies_test = pd.get_dummies(x_test['Fam'])

x_test_no_Fam = x_test.drop(['Fam'], axis = 1)

x_test = pd.concat([x_test_no_Fam, Fam_dummies_test], axis=1)
# Training dataset

Embarked_dummies_train = pd.get_dummies(x_train['Embarked'])

x_train_no_Embarked = x_train.drop(['Embarked'], axis = 1)

x_train = pd.concat([x_train_no_Embarked, Embarked_dummies_train], axis=1)

x_train
# Testing dataset

Embarked_dummies_test = pd.get_dummies(x_test['Embarked'])

x_test_no_Embarked = x_test.drop(['Embarked'], axis = 1)

x_test = pd.concat([x_test_no_Embarked, Embarked_dummies_test], axis=1)

x_test
# I will drop all numerical features from the main datasets

x_train_noNum = x_train.drop(['Age', 'Fare'], axis = 1)

x_test_noNum = x_test.drop(['Age', 'Fare'], axis = 1)
# That's my first Kernel! I will simply replace NaN with the median. Median is less sensitive to outliers.

age_train = x_train['Age']

age_train_filled = age_train.fillna(age_train.median())



age_test = x_test['Age']

age_test_filled = age_test.fillna(age_test.median())
# Check missing data in other numerical features

nan_train_fare = np.sum(x_train['Fare'].isnull())

# nan_train_Fam = np.sum(x_train['Fam'].isnull())

nan_test_fare = np.sum(x_test['Fare'].isnull())

# nan_test_Fam = np.sum(x_test['Fam'].isnull())



print('Number of NaN in \n','Train_Fare: ', nan_train_fare,'\n','Train_Fam: ', nan_train_Fam,'\n','Test_Fare: ', nan_test_fare,'\n','Test_Fam: ', nan_test_Fam,'\n')
# There is one missing value in the Fare column of the test data set. 

# I will replace with the median

fare_test = x_test['Fare']

fare_test_filled = fare_test.fillna(fare_test.median())
x_train_Num = pd.concat([age_train_filled, x_train['Fare']], axis = 1)

x_train_Num
x_test_Num = pd.concat([age_test_filled, fare_test_filled], axis = 1)

x_test_Num
# Import StandardScaler from sklearn

from sklearn.preprocessing import StandardScaler



# define scaler as an object

scaler = StandardScaler()

scaled_x_train_Num = scaler.fit_transform(x_train_Num)

scaled_x_test_Num = scaler.fit_transform(x_test_Num)
# Sklearn gives back an array. I create a dataframe of the standardized data

df_scaled_x_train_Num = pd.DataFrame(scaled_x_train_Num)

df_scaled_x_train_Num.columns = x_train_Num.columns.values

df_scaled_x_train_Num
# Sklearn gives back an array. I create a dataframe of the standardized data

df_scaled_x_test_Num = pd.DataFrame(scaled_x_test_Num)

df_scaled_x_test_Num.columns = x_test_Num.columns.values

df_scaled_x_test_Num
x_train_preprocessed = pd.concat([x_train_noNum, df_scaled_x_train_Num], axis = 1)

x_train_preprocessed
x_test_preprocessed = pd.concat([x_test_noNum, df_scaled_x_test_Num], axis = 1)

x_test_preprocessed
import tensorflow as tf

from sklearn.model_selection import train_test_split
## Split into test train and validation data

X_train, X_val, Y_train, Y_val = train_test_split(x_train_preprocessed, y_train, test_size=0.05, random_state=42)
# Model:

tf.keras.backend.clear_session()

tf.random.set_seed(51)

np.random.seed(51)





model = tf.keras.models.Sequential([   

            tf.keras.layers.Dense(19, activation=tf.keras.layers.LeakyReLU(alpha=0.1),input_shape=[X_train.shape[1]]),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(38, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(19, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(1, activation='sigmoid')

                                    ])





lr_schedule = tf.keras.callbacks.LearningRateScheduler(

    lambda epoch: 1e-5 * 10**(epoch / 20))



optimizer = tf.keras.optimizers.Adam(lr=1e-5)



model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])



history = model.fit(X_train, Y_train, epochs=100, validation_data=[X_val, Y_val], callbacks=[lr_schedule], verbose=1)
import matplotlib.pyplot as plt



lrs = 1e-5 * (10 ** (np.arange(100) / 20))

plt.semilogx(lrs, history.history["loss"])

plt.axis([1e-5, 1,0, 1])
tf.keras.backend.clear_session()

tf.random.set_seed(42)

np.random.seed(42)





model = tf.keras.models.Sequential([

            tf.keras.layers.Dense(19, activation=tf.keras.layers.LeakyReLU(alpha=0.01),input_shape=[X_train.shape[1]]),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(38, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),

            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(19, activation='sigmoid'),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(1, activation='sigmoid')

                                    ])





optimizer = tf.keras.optimizers.Adam(lr=0.05)



reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,

                              patience=4, min_lr=1e-8, mode='auto', verbose=1)





model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])



print(model.summary())



history = model.fit(X_train, Y_train, epochs=50, shuffle=True, validation_data=[X_val, Y_val], callbacks=[reduce_lr], verbose=1)
import matplotlib.pyplot as plt



def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

  

plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
preds_test = model.predict_classes(x_test_preprocessed)

predictions = preds_test.reshape((-1,))



# Save test predictions to file

output = pd.DataFrame({'PassengerId': raw_data_testing['PassengerId'],

                       'Survived': predictions})

output.to_csv('submission.csv', index=False)
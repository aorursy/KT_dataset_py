# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import tensorflow as tf

from tensorflow import keras

from sklearn import preprocessing

from tensorflow.keras import regularizers

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# preprocessing all data we need



def data_preprocessing(path):

    # cleaning out names and getting preson's titles. Rare titles converting to 'rare'

    def names_to_titles(names):

        titles = pd.Series((n.split(",")[1].split(".")[0].replace(" ", "") for n in names))

        titles = titles.replace('Miss', 'Ms')

        title_list = set(titles)

        tmp = titles.tolist()

        for t in title_list:

            repeat_num = tmp.count(t)

            if (repeat_num < 6):

                titles = titles.replace(t, "rare")

        

        title_list_fixed = list(set(titles))

        for i, t in enumerate(titles):

            titles[i] = title_list_fixed.index(t)

        

        return titles

    

    

    data = pd.read_csv(path)

    # cleaning data

    data = data.replace(['C', 'S', 'Q'], [0, 1, 2])

    data = data.replace(['male', 'female'], [0, 1])

    data.Age = data.Age.fillna(value=data.Age.mean())

    data.Fare = data.Fare.fillna(value=data.Fare.median())

    # simple feature engineering

    data['Family'] = data.SibSp + data.Parch

    data['Titles'] = names_to_titles(data.Name)

    

    clean_data = data.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

    

    clean_data = clean_data.fillna(0)

    clean_data.Fare = (clean_data.Fare - clean_data.Fare.mean()) / clean_data.Fare.std()

    clean_data.Age = (clean_data.Age - clean_data.Age.mean()) / clean_data.Age.std()

    

    

    return clean_data









split_rate = .8



training_data = data_preprocessing('/kaggle/input/titanic/train.csv')

training_data = training_data.drop(['PassengerId'], axis=1)



x_test = data_preprocessing('/kaggle/input/titanic/test.csv')



test_ids = pd.DataFrame()

test_ids['PassengerId'] = x_test.PassengerId



x_test = x_test.drop(['PassengerId'], axis=1)



y = training_data.Survived



training = training_data.drop(['Survived'], axis=1)



my_slice = int(len(training_data)*split_rate)

x_train = training[:my_slice]

y_train = y[:my_slice]



x_val = training[my_slice:]

y_val = y[my_slice:]









dim_size = x_train.shape[1]

# model

model = keras.Sequential([

    keras.layers.Dense(20, input_dim=dim_size, activation='relu'),

    keras.layers.Dropout(.45),                                      

    keras.layers.Dense(1, activation='sigmoid')

])
# compiler

opt = keras.optimizers.RMSprop(learning_rate=0.01)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics='accuracy')
# history = model.fit(x=x_train, y=y_train, verbose=0, batch_size=35, epochs=200, validation_data=(x_val, y_val))

history = model.fit(x=x_train, y=y_train, verbose=0, batch_size=35, epochs=200)
# getting prediction

pred = model.predict(x_test, batch_size=35)
# forming dataframe for output and saving it

survived = (np.round(pred, decimals=0)).tolist()

survived_df = pd.DataFrame(survived)

result = pd.DataFrame({'PassengerId':test_ids['PassengerId'], 'Survived':survived_df[0]})

result = result.astype('int')

result.to_csv('/kaggle/working/my_survival_prediction.csv', index=False)

print ('Done')

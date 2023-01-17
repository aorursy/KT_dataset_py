%matplotlib inline



import pandas as pd

import numpy as np

import tensorflow as tf

import re, math



train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



full_data = pd.concat([train_data, test_data])
def get_title(name):

    title_search = re.search(" ([A-Za-z]+)\.", name)

    if title_search:

        title = title_search.group(1)

    else:

        title = ""

    if title in ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Master', 'Sir', 'Rev']:

        title = "Mr"

    elif title in ['Countess', 'Jonkheer', '']:

        title = "RARE"

    elif title in ['Dona', 'Lady', 'Ms']:

        title = "Mrs"

    elif title in ['Mlle', 'Mme']:

        title = 'Miss'

    return title



title_values = {'Mrs':0, 'Mr':1, 'Miss':2, 'RARE':3}

            

train_data['Title'] = train_data['Name'].apply(get_title)

test_data['Title'] = test_data['Name'].apply(get_title)

full_data['Title'] = full_data['Name'].apply(get_title)



train_data['Title_Value']  = train_data['Title'].apply(lambda x : title_values[x])

test_data['Title_Value']  = test_data['Title'].apply(lambda x : title_values[x])

full_data['Title_Value']  = full_data['Title'].apply(lambda x : title_values[x])
dist_data = {}

titles = ['Mrs', 'Mr', 'Miss', 'RARE']

for title in titles:

    _ages = full_data[full_data['Title'] == title]['Age']

    _ages = _ages[_ages.notnull()]

    dist_data[title] = {}

    dist_data[title]['mean'] = np.mean(_ages)

    dist_data[title]['std'] = np.std(_ages)

    

#######################################

# create column : New_Age

#######################################

def assemble_ages(dataset):

    age = dataset['Age']

    title = dataset['Title']

    if np.isnan(age):

        age = dist_data[title]['mean'] + (np.random.rand() - 0.5) * dist_data[title]['std']

    return int(age)



train_data['Age'] = train_data.apply(assemble_ages, axis=1)

test_data['Age'] = test_data.apply(assemble_ages, axis=1)
def assemble_family_size(dataset):

    parch = dataset['Parch']

    sibSp = dataset['SibSp']

    family_size = parch + sibSp + 1

    return int(family_size)



train_data['Family_Size'] = train_data.apply(assemble_family_size, axis=1)

test_data['Family_Size'] = test_data.apply(assemble_family_size, axis=1)
train_data['Fare'] = train_data['Fare'].apply(lambda x: 0 if math.isnan(x) else x)

test_data['Fare'] = test_data['Fare'].apply(lambda x: 0 if math.isnan(x) else x)
# Set 'Embarked' field to values based the labels

labels = full_data['Embarked'].value_counts()

labels_value = {}

for i in range(len(labels.index)):

    label = labels.index[i]

    labels_value[label] = i + 1



train_data['Embarked'] = train_data['Embarked'].apply(lambda x : 0 if pd.isnull(x) else labels_value[x])

test_data['Embarked'] = test_data['Embarked'].apply(lambda x : 0 if pd.isnull(x) else labels_value[x])
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test_data['Sex'] = test_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x : 0 if pd.isnull(x) else 1)

test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x : 0 if pd.isnull(x) else 1)
columns = ['Age','Embarked','Fare','Parch','SibSp','Pclass','Sex','Title_Value','Family_Size','Has_Cabin', 'Survived']



size = train_data.shape[0]

train_size = int(size * 0.8)



train_data_np = train_data.as_matrix(columns)

np.random.shuffle(train_data_np)



train_set = train_data_np[:train_size]

validate_set = train_data_np[train_size:]



train = train_set[:,:-1]

train_labels = train_set[:, [-1]]

validate = validate_set[:, :-1]

validate_labels = validate_set[:, [-1]]

test  = test_data.as_matrix(columns)[:, :-1]
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import RMSprop

from keras.regularizers import l2



batch_size = 20

num_epoch = 300

learning_rate = 0.0005



model = Sequential()

model.add(Dense(150, input_dim=10, activation='relu', bias=True, W_regularizer=l2(0.01)))

model.add(Dropout(0.5))

model.add(Dense(150, activation='relu', bias=True, W_regularizer=l2(0.01)))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid', bias=True, W_regularizer=l2(0.01)))



model.summary()



model.compile(optimizer=RMSprop(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train,train_labels,batch_size=batch_size,nb_epoch=num_epoch,verbose=0,validation_data=(validate, validate_labels))
import matplotlib.pyplot as plt



loss_and_acc = history.history

loss = loss_and_acc['loss']

acc = loss_and_acc['acc']

val_loss = loss_and_acc['val_loss']

val_acc = loss_and_acc['val_acc']

x_axis = range(1, num_epoch + 1)



fig = plt.gcf()

fig.set_figheight(4)

fig.set_figwidth(12)



loss_ax = plt.subplot(121)

lines = loss_ax.plot(x_axis, loss, 'r-', x_axis, val_loss, 'b-')

loss_ax.legend(lines, ('Train_Loss', 'Validation_Loss'), fontsize=10)

loss_ax.set_xlabel('Epoch')

loss_ax.set_ylabel('Loss')



acc_ax = plt.subplot(122)

lines = acc_ax.plot(x_axis, acc, 'r-', x_axis, val_acc, 'b-')

acc_ax.legend(lines, ('Train_Acc', 'Validation_Acc'), loc='lower right', fontsize=10)

acc_ax.set_xlabel('Epoch')

acc_ax.set_ylabel('Accuracy')

acc_ax.set_ylim((0, 1))

acc_ax.set_yticks(np.arange(0, 1.1, 0.1))



plt.show()
predictions = model.predict_classes(test, verbose=0)

passenger_ids = test_data['PassengerId']

predict_df = pd.DataFrame(predictions, passenger_ids, columns=['Survived'])

predict_df.to_csv("predictions.csv")
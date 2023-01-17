# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import re

import sklearn.metrics  as sklm

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset = dataset.fillna(np.nan)



dataset.head(15)
def find_title(name):

    result = re.search(r'[A-Z][a-z]*[.]', name)

    if (result == None):

        return ''

    return result.group(0)



dataset['Title'] = dataset.Name.apply(find_title)

dataset.Title.unique()
def map_title_to(title, _from, _to):

    if title == _from:

        return _to

    return title



dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Sir.','Mr.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Lady.','Mrs.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Dona.','Mrs.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Don.','Mr.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Capt.','Major.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Col.','Major.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Countess.','Mrs.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Jonkheer.','Mr.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Mlle.','Ms.'))

dataset.Title = dataset.Title.apply(lambda x: map_title_to(x, 'Mme.','Mrs.'))



dataset.Title.value_counts().plot(kind='bar')

plt.show()
def set_median_age_by_title(title):

    title_age_mask = dataset.loc[(dataset['Title'] == title) & (dataset['Age'].isna() )]

    df = dataset.iloc[title_age_mask.index]

    df.Age = dataset.Age.median()

    dataset.iloc[title_age_mask.index] = df



for title in dataset.Title.unique():

    set_median_age_by_title(title)
def set_age_interval(x):

    if x < 5:

        return 5

    if x < 10:

        return 10

    if x < 20:

        return 20

    if x < 30:

        return 30

    if x < 40:

        return 40

    if x < 50:

        return 50

    if x < 60:

        return 60

    return 70

     



dataset.Age = dataset.Age.apply(set_age_interval)

dataset.Age.value_counts().plot(kind='bar')

plt.show()
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

dataset.drop(['Name','Ticket'], axis=1, inplace=True)  

dataset.head()
fare_midean = dataset['Fare'].median()

values = {'Cabin':'Z','Embarked':'S','Fare':fare_midean}

dataset.fillna(value=values,inplace=True)

dataset['Cabin1'] = dataset['Cabin'].str[0]    

dataset.drop(['Cabin'], axis=1, inplace=True)    

    

#encode labels

dataset['Sex'] = LabelEncoder().fit_transform(dataset['Sex'].values)

dataset['Cabin1'] = LabelEncoder().fit_transform(dataset['Cabin1'].values)

dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'].values)

dataset['Title'] = LabelEncoder().fit_transform(dataset['Title'].values)



dataset.head()
working_columns = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','Cabin1','Title','SibSp','Parch']

columns_to_scale = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','Cabin1','Title','SibSp','Parch']



def scale_dataset(dataset):

    scalers = {}

    for column in columns_to_scale:

        scaler = StandardScaler()

        dataset[column] = scaler.fit_transform(dataset[column].values.reshape(-1, 1))

        scalers[column] = scaler

    return dataset, scalers



# Splitting into labels and data

dataset, scalers = scale_dataset(dataset)
train_set = dataset.iloc[:train.shape[0]]

test_set = dataset.iloc[train.shape[0]:]



train_set.tail()

test_set.head()
from sklearn.manifold import TSNE

tsne_results = TSNE(n_components=2).fit_transform(train_set[working_columns].values)



df_subset = pd.DataFrame()

df_subset['tsne-2d-one'] = tsne_results[:,0]

df_subset['tsne-2d-two'] = tsne_results[:,1]

df_subset['survived'] = targets = train_set['Survived'].values.astype(int)



plt.figure(figsize=(16,10))

sns.scatterplot(

    x="tsne-2d-one", 

    y="tsne-2d-two",

    hue="survived",

    data=df_subset,

    legend="full",

    alpha=0.5

)
from sklearn.neighbors import LocalOutlierFactor



columns = [c for c in train_set.columns if c not in ["Survived", "PassengerId", "Pclass", "Sex", "Cabin1", "Embarked", 'FamilySize']]



clf = LocalOutlierFactor(n_neighbors=5, contamination=0.5)

y_pred = clf.fit_predict(train_set[columns].values)

X_scores = clf.negative_outlier_factor_



clf_df = pd.DataFrame( columns=['X_scores', 'y_pred'])

clf_df.X_scores = X_scores

clf_df.y_pred = y_pred

outlier_index = clf_df.index[clf_df['y_pred'] == -1].tolist()



train_set.drop(outlier_index)
y_source = train_set['Survived'].values.astype(int)

X_source = train_set[working_columns].values



X_test = test_set[working_columns].values

y_test = test_set['Survived'].values.astype(int)
g = sns.heatmap(train_set[working_columns].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size = 0.2)



print('X_train: {}'.format(X_train.shape))

print('y_train: {}'.format(y_train.shape))

print('X_test: {}'.format(X_test.shape))

print('y_test: {}'.format(y_test.shape))
from keras.callbacks import EarlyStopping

from keras import optimizers

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.regularizers import l2



def create_model2(optimizer, units = 16, init = 'glorot_uniform', activation = 'relu', l2_init = 0.001, dropout=0.1):    

    model = Sequential()

    model.add(Dense(units=units, input_dim=10, activation=activation, kernel_initializer=init, kernel_regularizer=l2(l2_init)))

    model.add(Dropout(dropout))

    #model.add(BatchNormalization())

    model.add(Dense(units=units, activation=activation, kernel_initializer=init, kernel_regularizer=l2(l2_init)))

    model.add(BatchNormalization())

    model.add(Dropout(dropout))

    model.add(Dense(units=units, activation=activation, kernel_initializer=init))

    #model.add(BatchNormalization())

    #model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid',kernel_initializer=init))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

    return model
#from keras.wrappers.scikit_learn import KerasClassifier

#from sklearn.model_selection import GridSearchCV



#estimator = KerasClassifier(build_fn=create_model2, verbose=0)



# grid search epochs, batch size and optimizer

#optimizers = [ 'adam']

#init = ['glorot_uniform']

#activations = ['relu']

#epochs = [400]

#batches = [64]

#l2_inits = [1e-1, 1e-2, 1e-3, 3e-4]

#dropouts = [0.1]

#hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init, activation=activations, l2_init=l2_inits, dropout=dropouts)

#grid = GridSearchCV(estimator=estimator, param_grid=hyperparameters)

#grid_result = grid.fit(X_train, y_train)



# summarize results

#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#means = grid_result.cv_results_['mean_test_score']

#stds = grid_result.cv_results_['std_test_score']

#params = grid_result.cv_results_['params']

#for mean, stdev, param in zip(means, stds, params):

#    print("%f (%f) with: %r" % (mean, stdev, param))
# grid search values

#values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

#all_train, all_test = list(), list()



#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



#for l2_param in values:

#    # define model

#    model = create_model2(optimizer=adam, l2_init=l2_param)

#    # fit model

#    model.fit(X_train, y_train, epochs=400, verbose=0)

#    # evaluate the model

#    _, train_acc = model.evaluate(X_train, y_train, verbose=0)

#    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

#    print('Param: %f, Train: %.3f, Test: %.3f' % (l2_param, train_acc, test_acc))

#    all_train.append(train_acc)

#    all_test.append(test_acc)
#'activation': 'relu', 'batch_size': 64, 'dropout': 0.1, 'epochs': 400, 'init': 'glorot_uniform', 'l2_init': 0.0003, 'optimizer': 'adam'}





#from numpy.random import seed

#seed(1)

#from tensorflow import set_random_seed

#set_random_seed(2)



callbacks = [EarlyStopping(monitor='val_loss', patience=100)]



#rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

#adamax = optimizers.Adamax(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

adam = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



best_model = create_model2(units=16, optimizer=adam, init='glorot_uniform', activation='relu', l2_init=0.3)

history = best_model.fit(X_train, y_train, 

                        validation_data=[X_test, y_test],

                        batch_size=64,

                        epochs=1000,

                        callbacks=callbacks,

                        verbose=1)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='center right')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='center right')

plt.show()
# evaluate the model

scores = best_model.evaluate(X_train, y_train, verbose=0)

print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))



scores = best_model.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))
predictions = best_model.predict_classes(test_set[working_columns].values)



submission = pd.DataFrame({

        "PassengerId": test_set["PassengerId"],

        "Survived": predictions.ravel() 

    })

submission.to_csv('submission.csv', index=False)
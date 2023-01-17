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

original =  pd.read_csv('../input/data.csv')

fi=original

fi=pd.DataFrame(fi)

fi = fi.drop(columns='Unnamed: 0')

fi = fi.drop(columns='ID')

fi = fi.drop(columns='Photo')

fi = fi.drop(columns='Flag')

fi = fi.drop(columns='Club Logo')

fi = fi.drop(columns='Joined')

#Correct currencies

curs=["Release Clause", "Value", "Wage"]

for cur in curs:

    

    def curr_value(x):

        x = str(x).replace('€', '')

        if('M' in str(x)):

            x = str(x).replace('M', '')

            x = float(x) * 1000000

        elif('K' in str(x)):

            x = str(x).replace('K', '')

            x = float(x) * 1000

        return float(x)

    fi[cur] = fi[cur].apply(curr_value)

   

#Correct -Dismiss + values

cols=["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW","LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM","CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB"]

for col in cols:

    fi[col]=fi[col].str[:-2]

    fi[col]=fi[col].astype(float)

    

#Convert contract end

fi['Contract Valid Until']=fi['Contract Valid Until'].str[-4:]

fi['Contract Valid Until']=fi['Contract Valid Until'].astype(float)

    

#Corect height values 

fi['Height']=fi['Height'].str.replace("'",'.')

fi['Height']=fi['Height'].astype(float)



#Correct Weight

fi['Weight']=fi['Weight'].str[:-3]

fi['Weight']=fi['Weight'].astype(float)



#X and y assignments

#fi=(fi[fi["Position"]!="GK"])

X = fi.loc[:, fi.columns != 'Value']

y=fi.loc[:,['Value']]

X = X.drop(columns='Name')

X = X.drop(columns='Real Face')

#identify Object columns

obj_df = X.select_dtypes(include=['object']).copy()

obj_df.head()

#Encoding -1

X_dum=pd.get_dummies(X[obj_df.columns], dummy_na=True,drop_first=True)

X = pd.concat([X.drop(obj_df.columns, axis=1), pd.get_dummies(X[obj_df.columns])], axis=1)

#See Correlations & Drop highly correlated attributes

X1 = pd.DataFrame(X)

corr = X1.corr()

corr_matrix = X1.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

X=X.drop(columns=to_drop, axis=1)

columnlist=X.columns.tolist()

# Taking care of missing data

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(X)

X_m = imputer.transform(X)

X=pd.DataFrame(X_m)



from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(y)

y = imputer.transform(y)

y=pd.DataFrame(y)



from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)

imputer = imputer.fit(y)

y = imputer.transform(y)

y=pd.DataFrame(y)



#Split Dataset Test vs. Train



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

def coeff_determination(y_test, y_pred):

    from keras import backend as K

    SS_res =  K.sum(K.square( y_test-y_pred ))

    SS_tot = K.sum(K.square( y_test - K.mean(y_test) ) )

    return ( 1 - SS_res/(SS_tot + K.epsilon()))
#deep learning with keras lib

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from keras import backend as K

# Initialising the ANN

regressor = Sequential()

# Adding the input layer and the first hidden layer

regressor.add(Dense(output_dim = 624, kernel_initializer='normal', activation = 'relu', input_dim = 1248))

# Adding the second hidden layer

regressor.add(Dense(output_dim = 300, kernel_initializer='normal', activation = 'relu'))

# Adding the 3rd hidden layer

regressor.add(Dense(output_dim = 150, kernel_initializer='normal', activation = 'relu'))

# Adding the 4th hidden layer

regressor.add(Dense(output_dim = 75, kernel_initializer='normal', activation = 'relu'))

# Adding the output layer

regressor.add(Dense(output_dim = 1, kernel_initializer='normal', activation = 'linear'))

# Compiling the ANN

def coeff_determination(y_test, y_pred):

    from keras import backend as K

    SS_res =  K.sum(K.square( y_test-y_pred ))

    SS_tot = K.sum(K.square( y_test - K.mean(y_test) ) )

    return ( 1 - SS_res/(SS_tot + K.epsilon()))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [coeff_determination])

# Fitting the ANN to the Training set-I have tried it with different batch sizes and epochs but it was overfitted

regressor.fit(X_train, y_train, batch_size = 950, nb_epoch = 7)
# Predicting the Test set results

y_pred = regressor.predict(X_test)
#Visualize Predicted vs. Actual

import matplotlib.pyplot as plt

_, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(x = range(0, y_test.size), y=y_test, c = 'green', label = 'Actual', alpha = 0.4)

ax.scatter(x = range(0, y_pred.size), y=y_pred, c = 'red', label = 'Predicted', alpha = 0.4)

plt.title('Actual vs. Predicted')

plt.xlabel('Test Size')

plt.ylabel('y Value')

def millions(x, pos):

    'The two args are the value and tick position'

    return '€%1.1fM' % (x * 1e-6)

ax.yaxis.set_major_formatter(plt.FuncFormatter(millions))

plt.legend()

plt.show()
""""# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 30)

print(accuracies)

print(accuracies.mean())

print(accuracies.std())"""""
#Avoid data type error

y_pred = y_pred.round().astype(int)

y[0] = y[0].round().astype(int)

#K-Fold Corss Validation

from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

cvscores = []

for train, test in kfold.split(X, y):

  # create model

    model = Sequential()

    model.add(Dense(624, input_dim=1248, activation='relu'))

    model.add(Dense(300, activation='relu'))

    model.add(Dense(150, activation='relu'))

    model.add(Dense(75, activation='relu'))

    model.add(Dense(1, activation='linear'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[coeff_determination])

    # Fit the model

    model.fit(X_train, y_train, epochs=7, batch_size=950)

    # evaluate the model

    scores = model.evaluate(X_test, y_test, verbose=0)

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
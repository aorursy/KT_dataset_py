from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

import pandas as pd 

import numpy as np
data = pd.read_csv('../input/diabetes.csv')

data.describe()
def replaceZeroWithMean(column):

    if column == None:

        return

    

    print('Column: ', column)

    

    print('Number of zero entries: ', len(data.loc[data[column]==0, column]))

    

    column_values_non_zero = data.loc[data[column] != 0, column]

    mean = sum(column_values_non_zero)/len(column_values_non_zero)

    print('Mean: ', mean)

    data.loc[data[column] == 0 , column] = mean

    print('---------------------------------------')    
columns_zero_to_mean = ['Glucose', 'BloodPressure', 'BMI']



for column in columns_zero_to_mean:

    replaceZeroWithMean(column)
data.describe()[columns_zero_to_mean] 
data.rename(columns = {'Outcome': 'class'}, inplace=True)
data.dtypes
data_y = data['class']

data_X = data.drop(['class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y,

                                                    train_size=0.75, test_size=0.25)
tpot = TPOTClassifier(verbosity=1, max_time_mins=15) 

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_script.py')
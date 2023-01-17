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


from numpy import loadtxt







import plotly.offline as po

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns



from xgboost import XGBClassifier



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from collections import Counter





def file_founder():

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))



            

def submission_file(test_data, test_preds):

    output = pd.DataFrame({'Id': test_data.index+15121,

                      'Cover_Type': test_preds})



    output.to_csv('submission.csv', index=False)

    

def feature_importances(model, X, y, figsize=(18, 6)):

    model = model.fit(X, y)

    

    importances = pd.DataFrame({'Features': X.columns, 

                                'Importances': model.feature_importances_})

    

    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)



    fig = plt.figure(figsize=figsize)

    sns.barplot(x='Features', y='Importances', data=importances)

    print(model.feature_importances_)

    plt.xticks(rotation='vertical')

    plt.show()

    return importances



# create a dict that map soil type with rockness

# 0=unknow 1=complex 2=rubbly, 3=stony, 

# 4=very stony, 5=extremely stony 6=extremely bouldery

soils = [

    [7, 15, 8, 14, 16, 17,

     19, 20, 21, 23], #unknow and complex 

    [3, 4, 5, 10, 11, 13],   # rubbly

    [6, 12],    # stony

    [2, 9, 18, 26],      # very stony

    [1, 24, 25, 27, 28, 29, 30,

     31, 32, 33, 34, 36, 37, 38, 

     39, 40, 22, 35], # extremely stony and bouldery

]



soil_dict = dict()

for index, values in enumerate(soils):

    for v in values:

        soil_dict[v] = index

        

        

def soil(df, soil_dict=soil_dict):

    df['Rocky'] =  sum(i * df['Soil_Type'+ str(i)] for i in range(1, 41))

    df['Rocky'] = df['Rocky'].map(soil_dict) 



    return df





def select(importances, edge):

    c = importances.Importances >= edge

    cols = importances[c].Features.values

    return cols





            

#def main():



train_dir = "/kaggle/input/learn-together/train.csv"

test_dir = "/kaggle/input/learn-together/test.csv"



train_df = pd.read_csv(train_dir)

test_df = pd.read_csv(test_dir)



array = train_df.values



X=train_df.copy()

TARGET='Cover_Type'

y=train_df[TARGET]

#y = array[:,-1]

#X = array[:,1:55]

X = soil(X)

test_df=soil(test_df)





# drop label 

if TARGET in X.columns:

    X.drop(TARGET, axis=1, inplace=True)



X.drop('Id', axis=1, inplace=True)

#seed = 7

#test_size = 0.33

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)



model = XGBClassifier()

importances = feature_importances(model, X, y)   

#model.fit(X_train, y_train)

col = select(importances, 0.003)



X = X[col]

test_df=test_df[col]

#print(test_df.describe())



#y_pred = model.predict(test_df)

#predictions = [round(value) for value in y_pred]



#submission_file(test_df, y_pred)



#     accuracy = accuracy_score(y_test, predictions)

#     print("Accuracy: %.2f%%" % (accuracy * 100.0))

    

    

#if __name__ == "__main__":

#    main()
model = model.fit(X, y)

y_pred = model.predict(test_df)

predictions = [round(value) for value in y_pred]



submission_file(test_df, y_pred)
print(X.shape)
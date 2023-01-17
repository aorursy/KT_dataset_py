from IPython.display import Image

Image("../input/ml-cplexity/ml.JPG")


import time

import math

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

class ComplexityEvaluator:

    def __init__(self , nrow_samples , ncol_samples):

        self._nrow_samples = nrow_samples

        self._ncol_samples = ncol_samples

    

    #random data

    def _time_samples(self , model , random_data_generator):

        row_list = []

        # iterate with rows and columns

        for nrow in self._nrow_samples:

            for ncol in self._ncol_samples:

                train , label = random_data_generator(nrow , ncol)

                #initiate timer

                start_time = time.time()

                model.fit(train , label)

                elapsed_time = time.time() - start_time

                result = {"N" : nrow , "P" : ncol , "Time" : elapsed_time}

                row_list.append(result)

                

        return row_list , len(row_list)

    

    #house pricing data

    def _time_houseprice(self , model):

        row_list = []

        #initiate timer

        train = self._nrow_samples

        label = self._ncol_samples

        start_time = time.time()

        model.fit(train , label)

        elapsed_time = time.time() - start_time

        #print("time : " , elapsed_time)

        result = {"N" :len(self._nrow_samples) , "P" : len(self._ncol_samples), "Time" : elapsed_time}

        row_list.append(result)

                

        return row_list , len(row_list)

    

    def run(self , model , random_data_generator , ds='random'):

        import random

        if ds == 'random':

            row_list , length = self._time_samples(model, random_data_generator)

        else:

            row_list , length = self._time_houseprice(model)

            

        cols = list(range(0 , length))

        data = pd.DataFrame(row_list , index =cols)

        print(data)

        data = data.applymap(math.log)

        #print("apply math : ", data)

        linear_model = LinearRegression(fit_intercept=True)

        linear_model.fit(data[["N" , "P"]] , data[["Time"]])

        #print("coefficients : " , linear_model.coef_)

        return linear_model.coef_

        
class TestModel:

    def __init__(self):

        pass

    

    def fit(self , x, y):

        time.sleep(x.shape[0] /1000)

        
def random_data_generator(n , p):

    return np.random.rand(n , p) , np.random.rand(n , 1)

if __name__ == "__main__":

    model = TestModel()

    nrow_samples = [200, 500, 1000, 2000, 3000]

    ncol_samples = [1,5,10]

    complexity_evaluator = ComplexityEvaluator(nrow_samples , ncol_samples)

    res = complexity_evaluator.run(model , random_data_generator)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.svm import SVR, SVC

from sklearn.linear_model import LogisticRegression
regression_models = [RandomForestRegressor(),

                     ExtraTreesRegressor(),

                     AdaBoostRegressor(),

                     LinearRegression(),

                     SVR()]



classification_models = [RandomForestClassifier(),

                         ExtraTreesClassifier(),

                         AdaBoostClassifier(),

                         SVC(),

                         LogisticRegression(),

                         LogisticRegression(solver='sag')]
names = ["RandomForestRegressor",

         "ExtraTreesRegressor",

         "AdaBoostRegressor",

         "LinearRegression",

         "SVR",

         "RandomForestClassifier",

         "ExtraTreesClassifier",

         "AdaBoostClassifier",

         "SVC",

         "LogisticRegression(solver=liblinear)",

         "LogisticRegression(solver=sag)"]
#using sample data to run on different models

sample_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

sample_data = sample_data.loc[:, sample_data.dtypes !=np.object]

sample_data = sample_data.fillna(0)

nrows = sample_data.iloc[:,:-1].values.tolist()

ncols = sample_data['SalePrice'].values.tolist()

complexity_evaluator = ComplexityEvaluator(nrows,ncols)
i = 0

for model in regression_models:

    res = complexity_evaluator.run(model, random_data_generator , 'houseprice')[0]

    print(names[i] + ' | ' + str(round(res[0], 2)) +

          ' | ' + str(round(res[1], 2)))

    i = i + 1
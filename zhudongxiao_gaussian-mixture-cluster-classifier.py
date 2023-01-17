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
import xgboost as xgb

from sklearn.mixture import GaussianMixture



class Data_clean(object):

    def __init__(self, trainPath, labelPath, submit_trainPath):

        #Read the raw data

        self.train = pd.read_csv(trainPath, header=None)

        self.label = pd.read_csv(labelPath, header=None).values.ravel()

        self.submit_train = pd.read_csv(submit_trainPath, header=None)



    def gaussian_cluster(self):

        #Do the gaussian mixture cluster on train data

        all_train = np.r_[self.train, self.submit_train]

        covType = ['spherical', 'tied', 'diag', 'full']

        n_component = [i for i in range(2, 11)]

        bestBIC = np.infty

        best_n, best_cov = 0, 'spherical'

        for cov in covType:

            for n in n_component:

                model = GaussianMixture(n_components=n, covariance_type=cov, random_state=random_seed)

                model.fit(all_train)

                _bic = model.aic(all_train)     #using aic can have a great benefit can bic for this case

                if _bic < bestBIC:

                    bestBIC = _bic

                    bestModel = model

                    best_n = n

                    best_cov = cov

        self.train = bestModel.predict(self.train).reshape(-1, 1)

        self.submit_train = bestModel.predict(self.submit_train).reshape(-1, 1)



    def run_gauss(self):

        self.gaussian_cluster()

        return self.train ,self.label, self.submit_train



class Classifier_Model(object):

    def __init__(self, xtrain, ytrain, submit_train, model):

        self.xtrain = xtrain

        self.ytrain = ytrain

        self.submit_train = submit_train

        self.model = model



    def model_predict(self):

        self.model.fit(self.xtrain, self.ytrain)

        ypred = self.model.predict(self.submit_train)

        ypred = pd.DataFrame(ypred)

        ypred.index += 1

        ypred.columns = ['Solution']

        ypred['Id'] = np.arange(1, ypred.shape[0] + 1)

        ypred = ypred[['Id', 'Solution']]

        ypred.to_csv('Submission_GassianCluster_XGB.csv', index = False)



    def run(self):

        self.model_predict()



if __name__ == '__main__':

    global random_seed

    random_seed = 9

    trainPath = '../input/train.csv'

    labelPath = '../input/trainLabels.csv'

    submit_trainPath = '../input/test.csv'



    data_clean = Data_clean(trainPath, labelPath, submit_trainPath)

    #In this case, Gaussian Mixture cluster does a great improvement for accuracy

    train, label, submit_train = data_clean.run_gauss()



    #In this case, many classifier can have a high accuracy, XGBClassifier is only a choose

    xgbboost = xgb.XGBClassifier(random_state=random_seed, min_samples_leaf=1, subsample=0.8, colsample_bytree=0.8,

                                 objective='binary:logistic', max_depth=10, n_estimators=150)

    classifier_model = Classifier_Model(train, label, submit_train, xgbboost)

    classifier_model.run()
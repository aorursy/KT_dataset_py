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
import numpy as np

import pandas as pd
import os

print(os.listdir("../input"))
from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import cohen_kappa_score

Input_path = '../input/train/'

data = pd.read_csv(Input_path +"train.csv", na_values=['no info','.'])
data = data.dropna(how ='any',axis=0)
dt1 = data.drop(['Name','State','RescuerID','Description','PetID'],axis=1)
dtyp = dt1.dtypes.astype(str).to_dict()

print(dtyp)
Ipath = '../input/test/'

datatest = pd.read_csv(Ipath +"test.csv", na_values=['no info','.'])
datatest1 = datatest.drop(['Name','State','RescuerID','Description','PetID'],axis=1)

dtyptest = datatest1.dtypes.astype(str).to_dict()
X_attributes=dt1.iloc[:,0:18]
y_target=dt1.iloc[:,18]

Classification = RandomForestClassifier()
forestgrid = {

    

    'bootstrap': [True],

    'max_depth': [None],

    'max_features': ['auto'],

    'min_samples_leaf': [5,10,15],

    'min_samples_split': [5,10,15],

    'n_estimators': [50,100,150]

}
gridoper = GridSearchCV(estimator = Classification, 

                           param_grid = forestgrid, 

                           cv = 7, 

                           verbose = 1,

                           n_jobs = -1)
gridoper.fit(X_attributes, y_target)
print("*********************************************")

print('Best parameters:', gridoper.best_params_)

print("*********************************************")

print('Quadratic weighted kappa score: ', cohen_kappa_score(gridoper.predict(X_attributes), 

                                y_target, weights='quadratic'))

print('***********Predictions on train data*********')



y_pred = gridoper.predict(X_attributes)

print('***********Predictions on train data*********')

y_predtestn = gridoper.predict(datatest1)



submission_df = pd.DataFrame(data={"PetID":datatest["PetID"], 

                                   "AdoptionSpeed": y_predtestn})



submission_df.head()

submission_df.to_csv('submission.csv',index=False)

print("************submission*************")

print(submission_df)



print("************submission*************")
'''

writepath = '../input/test/'

wr = open(writepath+"submission.csv", "w")

wr.write(str(submission_df))      

wr.close()

'''
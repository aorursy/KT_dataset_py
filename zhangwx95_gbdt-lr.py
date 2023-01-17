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
train_df = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

test_df =  pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')



###############################################################

#特征来源：https://www.kaggle.com/aharless/xgboost-cv-lb-284 

#然后发现内存吃不消。。。。

##############################################################

'''

NUMERIC_COLS = [

   "ps_car_13",  #            : 1571.65 / shadow  609.23

	"ps_reg_03",  #            : 1408.42 / shadow  511.15

	"ps_ind_05_cat",  #        : 1387.87 / shadow   84.72

	"ps_ind_03",  #            : 1219.47 / shadow  230.55

	"ps_ind_15",  #            :  922.18 / shadow  242.00

	"ps_reg_02",  #            :  920.65 / shadow  267.50

	"ps_car_14",  #            :  798.48 / shadow  549.58

	"ps_car_12",  #            :  731.93 / shadow  293.62

	"ps_car_01_cat",  #        :  698.07 / shadow  178.72

	"ps_car_07_cat",  #        :  694.53 / shadow   36.35

	"ps_ind_17_bin",  #        :  620.77 / shadow   23.15

	"ps_car_03_cat",  #        :  611.73 / shadow   50.67

	"ps_reg_01",  #            :  598.60 / shadow  178.57

	"ps_car_15",  #            :  593.35 / shadow  226.43

	"ps_ind_01",  #            :  547.32 / shadow  154.58

	"ps_ind_16_bin",  #        :  475.37 / shadow   34.17

	"ps_ind_07_bin",  #        :  435.28 / shadow   28.92

	"ps_car_06_cat",  #        :  398.02 / shadow  212.43

	"ps_car_04_cat",  #        :  376.87 / shadow   76.98

	"ps_ind_06_bin",  #        :  370.97 / shadow   36.13

	"ps_car_09_cat",  #        :  214.12 / shadow   81.38

	"ps_car_02_cat",  #        :  203.03 / shadow   26.67

	"ps_ind_02_cat",  #        :  189.47 / shadow   65.68

	"ps_car_11",  #            :  173.28 / shadow   76.45

	"ps_car_05_cat",  #        :  172.75 / shadow   62.92

	"ps_calc_09",  #           :  169.13 / shadow  129.72

	"ps_calc_05",  #           :  148.83 / shadow  120.68

	"ps_ind_08_bin",  #        :  140.73 / shadow   27.63

	"ps_car_08_cat",  #        :  120.87 / shadow   28.82

	"ps_ind_09_bin",  #        :  113.92 / shadow   27.05

	"ps_ind_04_cat",  #        :  107.27 / shadow   37.43

	"ps_ind_18_bin",  #        :   77.42 / shadow   25.97

	"ps_ind_12_bin",  #        :   39.67 / shadow   15.52

	"ps_ind_14",  #            :   37.37 / shadow   16.65



]

'''







NUMERIC_COLS = ["ps_reg_01", "ps_reg_02", "ps_reg_03","ps_car_12", "ps_car_13", "ps_car_14", "ps_car_15"]

from sklearn.model_selection import train_test_split



X_train, X_test,y_train, y_test = train_test_split(train_df[NUMERIC_COLS], train_df['target'], test_size=0.3, random_state=2019)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from lightgbm.sklearn import LGBMClassifier

num_leaf = 64
gbm = LGBMClassifier(

            n_estimators = 20,

            objective = 'binary',

            boosting = 'gbdt',

            num_leaves = 64,

            random_state = 2020

        )



gbm.fit(X_train,y_train)
y_train_1 = y_train.reindex(range(len(y_train)),fill_value = 0)

print((y_train_1))
print('Writing transformed training data')



transformed_training_matrix = np.zeros([y_train_1.shape[0], 20 * num_leaf],dtype=np.int64)  # N * num_tress * num_leafs

for i in range(0, y_train_1.shape[0]):



    temp = np.arange(20) * num_leaf + np.array(y_train_1[i])

    transformed_training_matrix[i][temp] += 1

    

    

    

y_pred = gbm.predict(X_test)

print('Writing transformed testing data')

transformed_testing_matrix = np.zeros([y_pred.shape[0], 20 * num_leaf],dtype=np.int64)

for i in range(0,  y_pred.shape[0]):

    temp = np.arange(20) * num_leaf + np.array(y_pred[i])

    transformed_testing_matrix[i][temp] += 1
from sklearn.linear_model import LogisticRegression 

lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction

lm.fit(transformed_training_matrix,y_train)  # fitting the data

y_pred_test = lm.predict_proba(transformed_testing_matrix)   # Give the probabilty on each label
NE = (-1) / len(y_pred_test) * sum(((1+y_test)/2 * np.log(y_pred_test[:,1]) +  (1-y_test)/2 * np.log(1 - y_pred_test[:,1])))

print("Normalized Cross Entropy " + str(NE))
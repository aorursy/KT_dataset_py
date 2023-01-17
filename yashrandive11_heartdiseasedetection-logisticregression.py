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
import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()



from scipy import stats

stats.chisqprob = lambda chisq, df:stats.chi12.sf(chisq,df)
raw_data = pd.read_csv('../input/logistic-regression-heart-disease-prediction/framingham_heart_disease.csv')

raw_data.describe(include = 'all')
raw_data.isnull().sum()
data_no_mv = raw_data.dropna(axis=0)
data_no_mv.isnull().sum()
data = data_no_mv.drop(['education'], axis = 1)

data
y = data['TenYearCHD']

x1 = data.drop(['TenYearCHD'], axis = 1)
x = sm.add_constant(x1)

reg_log = sm.Logit(y,x)

results_log = reg_log.fit()

results_log.summary()
data_cleaned = x.copy()

data_cleaned = data_cleaned.drop(['BPMeds'], axis = 1)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train, y_test = train_test_split(data_cleaned, y, test_size=0.2, random_state = 365)
x_cleaned = x_train

x_new = sm.add_constant(x_cleaned)

reg_log = sm.Logit(y_train,x_new)

results_log = reg_log.fit()

results_log.summary()
def confusion_matrix(data,actual_values,model):

        

        # Confusion matrix 

        

        # Parameters

        # ----------

        # data: data frame or array

            # data is a data frame formatted in the same way as your input data (without the actual values)

            # e.g. const, var1, var2, etc. Order is very important!

        # actual_values: data frame or array

            # These are the actual values from the test_data

            # In the case of a logistic regression, it should be a single column with 0s and 1s

            

        # model: a LogitResults object

            # this is the variable where you have the fitted model 

            # e.g. results_log in this course

        # ----------



        #Predict the values using the Logit model

        pred_values = model.predict(data)

        # Specify the bins 

        bins=np.array([0,0.5,1])

        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0

        # if they are between 0.5 and 1, they will be considered 1

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        # Calculate the accuracy

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        # Return the confusion matrix and

        string = 'Accuracy is ' + repr(accuracy*100)+' %'

        return cm, string
confusion_matrix(x_new,y_train,results_log)
x_test1 = x_test

reg_log = sm.Logit(y_test,x_test)

results_log1 = reg_log.fit()

results_log1.summary()
def confusion_matrix(data,actual_values,model):

        

        # Confusion matrix 

        

        # Parameters

        # ----------

        # data: data frame or array

            # data is a data frame formatted in the same way as your input data (without the actual values)

            # e.g. const, var1, var2, etc. Order is very important!

        # actual_values: data frame or array

            # These are the actual values from the test_data

            # In the case of a logistic regression, it should be a single column with 0s and 1s

            

        # model: a LogitResults object

            # this is the variable where you have the fitted model 

            # e.g. results_log in this course

        # ----------



        #Predict the values using the Logit model

        pred_values = model.predict(data)

        # Specify the bins 

        bins=np.array([0,0.5,1])

        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0

        # if they are between 0.5 and 1, they will be considered 1

        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]

        # Calculate the accuracy

        accuracy = (cm[0,0]+cm[1,1])/cm.sum()

        # Return the confusion matrix and

        string = 'Accuracy is ' + repr(accuracy*100)+' %'

        return cm, string
confusion_matrix(x_new,y_train,results_log)
cm_df = pd.DataFrame(results_log.pred_table())

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0:'Actual 0',1:'Actual 1'})

cm_df
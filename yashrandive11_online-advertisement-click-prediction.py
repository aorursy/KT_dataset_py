import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()



from scipy import stats

stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
raw_data = pd.read_csv('../input/advertising.csv')

raw_data
raw_data.describe(include = 'all')
data = raw_data.copy()
data = data.drop(['Ad Topic Line','City', 'Country', 'Male','Timestamp'], axis = 1)

data
data.columns.values
y = data['Clicked on Ad']

x1 = data[['Daily Time Spent on Site', 'Age', 'Area Income',

       'Daily Internet Usage']]
plt.scatter(data['Daily Time Spent on Site'],y)

plt.xlabel('Daily Time Spent on Site', fontsize = 10)

plt.ylabel('Clicked on Ad', fontsize = 10)

plt.show()
plt.scatter(data['Age'],y)

plt.xlabel('Age', fontsize = 10)

plt.ylabel('Clicked on Ad', fontsize = 10)

plt.show()
plt.scatter(data['Area Income'],y)

plt.xlabel('Area Income', fontsize = 10)

plt.ylabel('Clicked on Ad', fontsize = 10)

plt.show()
plt.scatter(data['Daily Internet Usage'],y)

plt.xlabel('Daily Internet Usage', fontsize = 10)

plt.ylabel('Clicked on Ad', fontsize = 10)

plt.show()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x1, y,test_size = 0.2, random_state = 365)
x= sm.add_constant(x_test)

reg_log = sm.Logit(y_test,x)

log_results = reg_log.fit()

log_results.summary()
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

        df = pd.DataFrame(data = cm, columns =['Predicted','Actual'])

        print(f"Accuracy= {accuracy*100}%" )

        return cm, accuracy
confusion_matrix(x,y_test,log_results)
cm_df = pd.DataFrame(log_results.pred_table())

cm_df.columns = ['Predicted 0','Predicted 1']

cm_df = cm_df.rename(index={0:'Actual 0',1:'Actual 1'})

cm_df
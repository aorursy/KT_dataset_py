import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

sns.set()
raw_data = pd.read_csv("../input/diabetes-dataset/traning dataset diabetes.csv")

raw_data
y = raw_data['Outcome']

x1 = raw_data[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']]



x = sm.add_constant(x1)

logistic_reg = sm.Logit(y,x)

result = logistic_reg.fit()



result.summary()

np.set_printoptions(formatter = {'float': lambda x: "{0:0.2f}".format(x)})

result.predict()
np.array(raw_data['Outcome'])
result.pred_table()
confusion_matrix = pd.DataFrame(result.pred_table())

confusion_matrix.columns = ['Predicted No Diabetes', 'Predicted Diabetes']

confusion_matrix = confusion_matrix.rename(index = {0 : 'Actual No Diabetes', 1 : 'Actual Diabetes'})

confusion_matrix
cm = np.array(confusion_matrix)

training_accuracy = (cm[0,0] + cm[1,1])/ cm.sum()

training_accuracy
test_data = pd.read_csv("../input/diabetes-dataset/test dataset diabetes.csv")

test_data
test_cleaned = test_data['Outcome']

test_data = test_data.drop(['Outcome'], axis = 1)

test_data = sm.add_constant(test_data)
def confusion_matrix(data, actual_values, model):

    predicted_values = model.predict(data)

    bins = np.array ([0, 0.5, 1])

    cm = np.histogram2d(actual_values, predicted_values, bins = bins)[0]

    accuracy = (cm[0,0] + cm[1,1])/cm.sum()

    return cm, accuracy
conf_matrix = confusion_matrix(test_data, test_cleaned, result)

conf_matrix
confusion_matrix = pd.DataFrame(conf_matrix[0])

confusion_matrix.columns = ['Predicted No Diabetes', 'Predicted Diabetes']

confusion_matrix = confusion_matrix.rename(index = {0 : 'Actual No Diabetes', 1 : 'Actual Diabetes'})

confusion_matrix
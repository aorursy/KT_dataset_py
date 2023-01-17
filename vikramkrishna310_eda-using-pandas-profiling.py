#importing necessary libraries and loading datasets into dataframes



import pandas as pd

dataset = pd.read_csv('../input/graduate-admissions/Admission_Predict.csv')
#checking wether data is loaded or not.



dataset.head()
#Describing the data



dataset.describe()
#Importing Pandas profiling for EDA

from pandas_profiling import ProfileReport

profile = ProfileReport(dataset)
profile
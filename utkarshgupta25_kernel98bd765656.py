import pandas as pd

import numpy as np
df = pd.read_csv("../input/Training Data - Classification of Patients with Abnormal Blood Pressure (N2000)_27-Jul-2016.csv")



df.head()



df.isna().sum()
df['Genetic_Pedigree_Coefficient'].fillna(df['Genetic_Pedigree_Coefficient'].mean(), inplace = True)

df['alcohol_consumption_per_day'].fillna(df['alcohol_consumption_per_day'].mean(), inplace = True)
df.drop(columns = ['Pregnancy'], inplace = True)
import seaborn as sns

import matplotlib.pyplot as plt



plt.hist(df['alcohol_consumption_per_day'])

plt.show()

plt.hist(df['Genetic_Pedigree_Coefficient'])

plt.show()
df[df.drop(columns = ['Patient_Number'], axis = 1).columns].hist(figsize = (16,16))

    
def plotBarChart(data,col,label):

    g = sns.FacetGrid(data, col=col)

    g.map(plt.hist, label, bins=10)



for val in ['Genetic_Pedigree_Coefficient','Level_of_Hemoglobin','Chronic_kidney_disease','Adrenal_and_thyroid_disorders','Age','BMI','Physical_activity', "Smoking"]:

    plotBarChart(df,'Blood_Pressure_Abnormality',val)
df['Normal'] = 0

df['Underweight'] = 0

df['Overweight'] = 0

df['Obese'] = 0



df['Underweight'] = df['BMI'].apply(lambda x: 1 if x < 19 else 0)

df['Normal'] = df['BMI'].apply(lambda x: 1 if x >= 19 and x <= 25 else 0)

df['Overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 and x <= 30 else 0)

df['Obese'] = df['BMI'].apply(lambda x: 1 if x > 30 else 0)
for val in ['Underweight', 'Normal', 'Overweight', 'Obese']:

    plotBarChart(df, 'Blood_Pressure_Abnormality', val)
df['Normal_haemo'] = df['Level_of_Hemoglobin'].apply(lambda x: 1 if x >= 10 and x <= 13 else 0)
plotBarChart(df, 'Blood_Pressure_Abnormality', 'Normal_haemo')
from sklearn.model_selection import train_test_split as tts

from sklearn.linear_model import LogisticRegression as lr

from sklearn import metrics



regressor = lr()



features = ['Genetic_Pedigree_Coefficient','Normal_haemo','Chronic_kidney_disease','Adrenal_and_thyroid_disorders', 'BMI', 'Age']



x = df[features]

y = df['Blood_Pressure_Abnormality']



x_train, x_test, y_train, y_test = tts(x,y)



regressor.fit(x_train, y_train)



predicted_values = regressor.predict(x_test)



print('Training data score:', regressor.score(x_train, y_train))



print('Testing data score:', regressor.score(x_test, y_test))
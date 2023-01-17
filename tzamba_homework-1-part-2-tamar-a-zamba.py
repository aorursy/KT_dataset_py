# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt #for plotting matrix
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Prepare data (Load, Clean, Inspect)
import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
#Load the dataset into a dataframe
df = pd.read_csv("../input/diabetes.csv")
#Load and review the data
pd.set_option("display.max_rows",768)
pd.set_option("display.max_columns",9)

#Clean: Check for null values
df.dropna(thresh=1)
#nspect: Visualize correlation
url = "../input/diabetes.csv"
data = pd.read_csv(url)
correlations = df.corr()
# plot matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
plt.show()
#Clean: Delete the skin column.
df = data.drop('SkinThickness',1)

#The  true and false vaue were already set to  1's and 0's.
#Check True/False ratio
true  = (df['Outcome']==1).sum()
false = (df['Outcome']==0).sum()
percentage = 100*(true/(true+false))
print (percentage)
#Split the data into training data and test data
x= df.values[:, 0:6]
y= df.values[:,7]
x_train, x_test, y_train,y_test = train_test_split (x,y,test_size=0.3, random_state = 100)
df= df[(df[['Glucose','BloodPressure','Insulin', 'BMI','DiabetesPedigreeFunction','Age']] != 0).all(axis=1)]
df
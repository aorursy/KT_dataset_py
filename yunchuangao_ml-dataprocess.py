import pandas as pd
import numpy as np
import os
# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
# Display the folders and files in current directory;
import os
for dirname, _, filenames in os.walk('/kaggle/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load data
df_train = pd.read_csv('/kaggle/input/titanic-survival-data/titanic_data.csv')
# Show first lines of data
df_train.head()
df_train.shape
df_train.info()
df_train.describe()
# Function to check the missing percent of a DatFrame;
def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
check_missing_data(df_train)
# Missing data: Cabin has high rate of missing data; insted of deleting the column,
# I will give 1 if Cabin is not null; otherwise 0;
df_train['Cabin']=np.where(df_train['Cabin'].isnull(),0,1)
# Combine train and test data, fill the missing values;
dataset = [df_train]

# def missing_data(x):
for data in dataset:
    #complete missing age with median
    data['Age'].fillna(data['Age'].mean(), inplace = True)

    #complete missing Embarked with Mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

    #complete missing Fare with median
    data['Fare'].fillna(data['Fare'].mean(), inplace = True)
check_missing_data(df_train)
df_train.head()
# Delete the irrelavent columns: Name, Ticket (which is ticket code)
drop_column = ['Name','Ticket','Embarked']
df_train.drop(drop_column, axis= 1, inplace = True)
df_train.head()
df_train.corr()
df_train.std()
# Convert ‘Sex’ feature into numeric.
genders = {"male": 0, "female": 1}
all_data = [df_train]

for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(genders)
df_train['Sex'].value_counts()
df_train.head()
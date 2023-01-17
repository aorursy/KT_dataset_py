# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
df=pd.read_csv("/kaggle/input/loan-data-set/loan_data_set.csv")
df
df.head()
df.tail()
df.shape
df.mean()
df.median()
df.mode()
df.describe()
df.corr()
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + "columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              "columns that have missing values")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
missing_values_table(df)
numerical_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term","Credit_History"]
from matplotlib import pyplot as plt
df[numerical_cols].hist(figsize=(30,15), bins=20)
plt.suptitle("Histograms of numerical values")
plt.show()
df.corr()
import seaborn as sns
sns.heatmap(df.corr())
comparison_column = np.where(df["ApplicantIncome"] == df["LoanAmount"], True, False)
df["equal"] = comparison_column
df


df.plot.hist()

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
# Use pd.read_csv to read file
path = "../input/star-categorization-giants-and-dwarfs/Star99999_raw.csv"
raw_data = pd.read_csv(path)

raw_data
raw_data.columns
# read some statistics of the dataset
raw_data.describe()
# Check the DataType of our dataset
raw_data.info()
# Convert Columns data type to float values
raw_data["Vmag"] = pd.to_numeric(raw_data["Vmag"], downcast="float", errors='coerce')
raw_data["Plx"] = pd.to_numeric(raw_data["Plx"], downcast="float", errors='coerce')
raw_data["e_Plx"] = pd.to_numeric(raw_data["e_Plx"], downcast="float", errors='coerce')
raw_data["B-V"] = pd.to_numeric(raw_data["B-V"], downcast="float", errors='coerce')
# Check the DataType of our dataset
raw_data.info()
# Actually , if you want to show all the columns you can add parameter `include='all'`.
raw_data.describe(include='all')
# get the number of missing data points per column
missing_values_count = raw_data.isnull().sum()

missing_values_count
# how many total missing values do we have?
total_cells = np.product(raw_data.shape)
total_missing = missing_values_count.sum()

# percentage of data that is missing
percent_missing = (total_missing/total_cells)
print("Percentage Missing:", "{:.2%}".format(percent_missing))
# remove all the rows that contain a missing value
# better to store it into a new variable to avoid confusion
raw_data_na_dropped = raw_data.dropna() 

raw_data_na_dropped
# just how much rows did we drop?
dropped_rows_count = raw_data.shape[0]-raw_data_na_dropped.shape[0]
print("Rows we dropped from original dataset: %d \n" % dropped_rows_count)

# Percentage we dropped
percent_dropped = dropped_rows_count/raw_data.shape[0]
print("Percentage Loss:", "{:.2%}".format(percent_dropped))
raw_data_na_dropped.describe()
#The best way to do this in pandas is to use drop:
raw_data_na_dropped = raw_data_na_dropped.drop('Unnamed: 0', axis=1)
raw_data_na_dropped.describe()
raw_data_na_dropped.info()
raw_data_na_dropped_reindex = raw_data_na_dropped.reset_index(drop=True)
raw_data_na_dropped_reindex.info()
#Optional - Save our progress
#raw_data_na_dropped_reindex.to_csv("Star99999_na_dropped.csv", index=False)
#Save a copy, so I can call it in a easier way
df = raw_data_na_dropped_reindex.copy()
df
#Dropping rows that `Plx` = 0
df = df[df.Plx != 0]

#Reindex the dataframe
df = df.reset_index(drop=True)

df
#Implement the equation
df["Amag"] = df["Vmag"] + 5* (np.log10(abs(df["Plx"]))+1)

df
df.info()
df.describe()
# Take a look at our SpType column
df['SpType']
#Copy the SpType column to a new column called TargetClass
df['TargetClass'] = df['SpType']

df
#The intuitive approach (Could take a long time if you have a huge dataset)
for i in range(len(df['TargetClass'])):
    if "V" in df.loc[i,'TargetClass']: 
        if "VII" in df.loc[i,'TargetClass']: 
            df.loc[i,'TargetClass'] = 0 # VII is Dwarf
        else:
            df.loc[i,'TargetClass'] = 1 # IV, V, VI are Giants
    elif "I" in df.loc[i,'TargetClass']: 
        df.loc[i,'TargetClass'] = 0 # I, II, III are Dwarfs
    else: 
        df.loc[i,'TargetClass'] = 9 # None
        
df['TargetClass']
df.describe(include='all')
#Save our progress
#df.to_csv("Star99999_preprocessed0821.csv", index=False)
df['TargetClass'].value_counts()
import matplotlib.pyplot as plt # plot graphs
import seaborn as sns # plot graphs

sns.countplot(df['TargetClass'])
#Dropping rows that `TargetClass` = 9
df = df[df.TargetClass != 9]

#Reindex the dataframe
df = df.reset_index(drop=True)

df
# Separate the labels
df_giants = df[df.TargetClass == 1]
df_dwarfs = df[df.TargetClass == 0]
# Numbers of rows of Giants and Dwarfs
num_of_giant = df_giants.shape[0]
num_of_dwarf = df_dwarfs.shape[0]
print("Giants(1):",num_of_giant)
print("Dwarfs(0):",num_of_dwarf)
from sklearn.utils import resample
# Downsample majority class
df_giants_downsampled = resample(df_giants, 
                                 replace=False,    # sample without replacement
                                 n_samples=num_of_dwarf,     # to match minority class
                                 random_state=1) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_giants_downsampled, df_dwarfs])
df_downsampled['TargetClass'].value_counts()
sns.countplot(df_downsampled['TargetClass'])
df_downsampled.describe(include='all')
df_downsampled.info()
df_balanced = df_downsampled.reset_index(drop=True)

df_balanced.info()
df_balanced
df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

df_balanced
#Save our dataset, we can finally play with it!!!
df_balanced.to_csv("Star39552_balanced.csv", index=False)
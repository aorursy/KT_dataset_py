# Exploratory data analysis of rockyou.txt

# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import hashlib

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
%%time
#dataframe = pd.read_fwf('../input/rockyou.txt', widths=24)

dataframe = pd.read_csv('../input/rockyou.txt',
                        delimiter = "\n", 
                        header = None, 
                        names = ["Passwords"],
                        encoding = "ISO-8859-1")
%%time
# dataframe.to_csv('../input/rockyou.csv')
# OSError: [Errno 30] Read-only file system: '../input/rockyou.csv'
dataframe.info()
dataframe.head()
%%time
dataframe['MD5'] = [hashlib.md5(str.encode(str(i))).hexdigest() 
                    for i in dataframe['Passwords'].fillna(0).astype(str)]
dataframe.head()
dataframe.info()
# Drop duplicate password
dataframe.drop_duplicates(subset=['Passwords'], keep=False, inplace=True)
dataframe.info()
%%time
## dataframe['Passwords'].value_counts()
# get indexes
## dataframe['Passwords'].value_counts().index.tolist()
# get values of occurrences
## dataframe['Passwords'].value_counts().values.tolist()
# delete all rows with password over 20 letters and less than 3
clutter = dataframe[ (dataframe['Passwords'].str.len() >= 20) 
                   | (dataframe['Passwords'].str.len() <= 3) ].index
dataframe.drop(clutter, inplace=True)
print (dataframe['Passwords'].str.len().value_counts())
dataframe.info()
dataframe = dataframe.reset_index(drop=True)
dataframe.info()

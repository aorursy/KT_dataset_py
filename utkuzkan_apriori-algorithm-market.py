# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules 

import random



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
orders =pd.read_csv("/kaggle/input/ecommerce-data/data.csv", encoding="windows-1252")
orders
#Şube Kodu 32 olanları filtrele

filter1=orders["Country"]=="France"

branch = orders[filter1]
branch
#Dataframe içinde NAN data NA data var mı diye kontrol yapıyoruz

branch.columns[branch.isnull().any()]
branch["Description"].value_counts()
# Stripping extra spaces in the description 

branch['Description'] = branch['Description'].str.strip() 
# Dropping the rows without any invoice number 

branch.dropna(subset =['InvoiceNo'], inplace = True) 

branch['InvoiceNo'] = branch['InvoiceNo'].astype('str') 
# Dropping all transactions which were done on credit 

branch = branch[~branch['InvoiceNo'].str.contains('C')] 
dict = {}

for index,row in branch.iterrows():

    key=row["InvoiceNo"]   

    #check key exists or not

    if key not in dict:

        dict[row["InvoiceNo"]] = []

        #append value

        dict[row["InvoiceNo"]].append(row["Description"])

    else:

        dict[row["InvoiceNo"]].append(row["Description"])

dict
list(branch["Description"].unique())
new_df = pd.DataFrame(columns=list(branch["Description"].unique()))
new_df
#add encoding values to new DataFrame

for key, value in dict.items():

    temp=[]

    for column in new_df.columns:

            for i in value:

                if i == column:

                    temp.append(column)

    encoded_rows = [] 

    for column in new_df.columns:

        if column in temp:

            encoded_rows.append(1)

        else:

            encoded_rows.append(0)

    new_df=new_df.append(pd.Series(encoded_rows, index=new_df.columns), ignore_index=True)

    



del dict    
new_df
# Building the model 

frq_items = apriori(new_df, min_support = 0.05, use_colnames = True) 

  

# Collecting the inferred rules in a dataframe 

rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 

rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False]) 

rules.head()
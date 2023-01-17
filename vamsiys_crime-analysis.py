# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_dict = pd.read_excel('../input/combined-data/Output.xls', sheet_name=None)
data_dict.keys()
id_cols = ["Area_Name", "Year", "Group_Name", "Sub_Group_Name", "Subgroup"]
modified_dict = {}
for key, value in data_dict.items():
    modified_dict[key] = value.melt(id_vars=value.columns.intersection(id_cols), var_name="Type", value_name="Count")
df_combined = pd.concat(modified_dict, axis=0, join='outer')
df_combined.head()
#data = pd.read_csv('../input/39_Specific_purpose_of_kidnapping_and_abduction.csv')
#data.head()
#id_cols = ['Area_Name', 'Year', 'Group_Name', 'Sub_Group_Name', 'K_A_Cases_Reported']
#maha = data[data['Area_Name'] == "Maharashtra"]
#maha.head()
#maha.loc[:, maha.columns.difference(id_cols)] = maha[maha.columns.difference(id_cols)].apply(lambda x: x.astype('float'))
#maha['Total_cases'] = maha[maha.columns.difference(id_cols)].sum(axis=1)
#maha.head()
#maha_summary = maha.groupby(by='Year').sum()
#maha_summary.to_csv('Maharashtra_kidnappings.csv')
#data_rape = pd.read_csv('../input/20_Victims_of_rape.csv')
#data_rape.head()
#data_rape_clean = data_rape[data_rape['Subgroup'] == "Total Rape Victims"]
data_rape_clean.groupby('Area_Name')['Victims_of_Rape_Total'].sum()

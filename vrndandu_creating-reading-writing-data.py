import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
df = pd.DataFrame({'Apples': 30,'Bananas': pd.Series(21, index=list(range(1))) })
check_q1(df)
idx = ('2017 Sales','2018 Sales')
df2 = pd.DataFrame({'Apples':(35, 41),'Bananas': pd.Series((21,34), index=idx)})
check_q2(df2)
idm = ('Flour','Milk','Eggs','Spam')
df3 = pd.Series(['4 cups','1 cup', '2 large','1 can'], index = idm).rename("Dinner")
check_q3(df3)
inputs = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
print(inputs.shape)
inputs.head()
in_xls = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheetname = 'Pregnant Women Participating')
in_xls.head()
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = q6_df.to_csv("cows_and_goats.csv")

import sqlite3 as sq
con = sq.connect("../input/pitchfork-data/database.sqlite")
df7 = pd.read_sql_query('SELECT * from artists',con)
print(df7.head())

con.close()

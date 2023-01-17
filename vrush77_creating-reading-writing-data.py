import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
data = [{'Apples':30,'Bananas': 21 }]
pd.DataFrame(data)
# Your code here
data = [{'Apples':35, 'Bananas':21, 'Apples':41,'Bananas': 34}]
pd.DataFrame(data, index = ['2017 Sales','2018 Sales'])
# Your code here
Data = {'Flour': '4 cups', 'Milk':'1 cup','Eggs':'2 large', 'Spam': '1 can'}
Dinner = pd.Series(Data)
Dinner = Dinner.rename('Dinner')
print(Dinner)
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
# Your code here
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', sheet_name = 'Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3
con = sqlite3.connect('../input/pitchfork-data/database.sqlite')
pd.read_sql_query('SELECT * FROM artists', con)
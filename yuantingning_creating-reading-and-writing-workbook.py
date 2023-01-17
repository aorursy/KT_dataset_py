import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())

pd.DataFrame({'Apple':[30],'Bananas':[21]})
# Your code here
pd.DataFrame({'Apple':[35,41],'Bananas':[21,34]},index=['2017 Sales','2018 Sales'])
# Your code here
pd.Series({'Flour':'4 cups','Milk':'1 cups','Eggs':'2 large','Spam':'1 can','Name:':'Dinner'})
print(answer_q3())
# Your code here 
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
# Your code here
print(answer_q5())
pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name='Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df

# Your code here
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
import sqlite3 
conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
pd.read_sql_query("SELECT*FROM artists",conn)
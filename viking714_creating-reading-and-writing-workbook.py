import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
# Your code here
# Your code here
Dinner = pd.Series({"Flour":"4 cups","Milk":"1 cup","Eggs":"2 large","Spam":"1 can"})
check_q3(Dinner)
# Your code here 
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
# Your code here
df5 = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls")
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df.to_csv("cows_and_goats.csv")
# Your Code Here
import sqlite3

con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
sql = "select * from artists"
df7 = pd.read_sql(sql,con)
df7.head()
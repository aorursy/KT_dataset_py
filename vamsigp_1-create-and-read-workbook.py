import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
a = [[30, 21]]
df = pd.DataFrame(data=a, columns=['Apples', 'Bananas'])
check_q1(df)
# Your code here
a = [[35, 21],[41,34]]
df2 = pd.DataFrame(data=a, columns=['Apples', 'Bananas'],index=['2017 Sales','2018 Sales'])
# print(df2)
check_q2(df2)
# Your code here
d = ['4 cups','1 cup','2 large', '1 can']
ind = ['Flour', 'Milk', 'Eggs','Spam']
ser = pd.Series(data=d, index=ind,name='Dinner')
# print(ser)
check_q3(ser)
# Your code here
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', header=0, index_col=0)
# print(df.head())
check_q4(df)
# Your code here
df1 = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
# df1.head()
check_q5(df1)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
filename = q6_df.to_csv('cows_and_goats.csv')
check_q6(filename)
# Your Code Here
import sqlite3

connection = sqlite3.connect('../input/pitchfork-data/database.sqlite')
tables = connection.execute("SELECT * FROM sqlite_master where type='table';")
# for t in tables:
#     print(t)
df7 = pd.read_sql_query("SELECT * from artists", connection)
# df.head()
check_q7(df7)
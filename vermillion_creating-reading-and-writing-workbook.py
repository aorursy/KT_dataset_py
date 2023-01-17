import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
pd.DataFrame({'Apples' : [30], 'Bananas' : [21]})
df = pd.DataFrame({'Apples' : [35,41], 'Bananas' : [21,34], 'Year':['2017 Sales','2018 Sales']})
df.set_index('Year')
pd.Series({"Flour":"4 cups",'Milk':'1 cup','Eggs':'2 large','Spam':'1 can'},name="Dinner")
pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
the_frame = pd.read_sql_query("SELECT * FROM %s;" % "../input/pitchfork-data/database-sqlite", engine)
pd.read_sql_table('../input/pitchfork-data/database-sqlite')
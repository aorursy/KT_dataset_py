import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
pd.DataFrame({'Apples': [30], 'Bananas': [21]})
# Your code here
df = pd.DataFrame({'Apples':[35, 41], 'Bananas':[21, 34]}, columns = ['Apples', 'Bananas'], index = ['2017 Sales', '2018 Sales'])
df
check_q2(df)
# Your code here
this_series = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index = ['Flour', 'Milk', 'Eggs', 'Spam'])
check_q3(this_series)
# Your code here 
read_in_csv = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv', index_col = 0)
check_q4(read_in_csv)
# Your code here
excel = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls', 'Pregnant Women Participating')
check_q5(excel)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df
# Your code here
check_q6(q6_df.to_csv('cows_and_goats.csv'))
# Your Code Here
from sqlalchemy import create_engine
engine = create_engine('sqlite:///../input/pitchfork-data/database.sqlite')
check_q7(pd.read_sql_table('artists', engine))

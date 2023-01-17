import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = pd.DataFrame([{'Apples':30,'Bananas':21}])
data
check_q1(data)
data = pd.DataFrame([{'Apples':35,'Bananas':21},{'Apples':41,'Bananas':34}], index = ['2017 Sales','2018 Sales'])
data
check_q2(data)
data = pd.Series({'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}, index = ['Flour','Milk','Eggs','Spam'])
data
check_q3(data)
data = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
data.columns
data = data[['country', 'description', 'designation', 'points',
       'price', 'province', 'region_1', 'region_2', 'variety', 'winery']]
check_q4(data)
data = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls", sheet_name = 'Pregnant Women Participating')
data
check_q5(data)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
cows_and_goats = q6_df.to_csv()
check_q6(cows_and_goats)
import sqlite3
con = sqlite3.connect("../input/pitchfork-data/database.sqlite")
data = pd.read_sql_query("select * from artists",con)
data
check_q7(data)
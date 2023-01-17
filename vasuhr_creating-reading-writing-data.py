import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame([{'Apples':30,'Bananas':21}]))
check_q2(pd.DataFrame([{'Apples':35,'Bananas':21},{'Apples':41,'Bananas':34}],index=['2017 Sales','2018 Sales']))
check_q3(pd.DataFrame({"Dinner":['4 cups','1 cup','2 large','1 can']},index=['Flour','Milk','Eggs','Spam'])["Dinner"])

check_q4(pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0))
check_q5(pd.read_excel(r"../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating"))



check_q6(pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2']))
check_q6(pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2']).to_csv("cows_and_goats.csv"))

# Your Code Here
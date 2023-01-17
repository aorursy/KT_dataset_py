import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
data = {"Apples": 30 ,"Bananas": 21}
dframe = pd.DataFrame(data,index=[0])
dframe


import numpy as np
import pandas as pd
from pandas import DataFrame

dict ={"Apples" : [35,21],"Bananas" :[21,34]}
dframe = DataFrame(dict,index = ["2017 Sales","2018 Sales"])
dframe
# Your code hereimport numpy as np
import pandas as pd
from pandas import Series

Dinner = Series(["4 cups","1 cup","2 large","1 can"],index = ["Flour","Milk","Eggs","Spam"])
Dinner
dframe = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
dframe
dframe = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls')
dframe
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats.csv')
# Your Code Here
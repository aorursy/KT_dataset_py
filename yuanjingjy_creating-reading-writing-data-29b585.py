import pandas as pd
pd.set_option('max_rows', 5)

import sys
sys.path.append('../input/advanced-pandas-exercises/')
from creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
import pandas as pd 
df_dictory={"Apples":[30],"Bananas":[21]}
df=pd.DataFrame(df_dictory)
df
# Your code here
import pandas as pd
import numpy as np
tmp=np.array([[35,21],[41,34]])
tmp
df=pd.DataFrame(tmp,columns=["Apples","Bananas"],index=["2017 Sales","2018 Sales"])
df
# Your code here
import pandas as pd 
tmp={"Flour":"4 cups","Milk":"1 cup","Eggs":"2  large","Spam":"1 can"}
Dinner=pd.Series(tmp)
Dinner
# Your code here 
data=pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")
data
# Your code here
df=pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls",sheet_name="Pregnant Women Participating")
df
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")
# Your Code Here
import sqlite3
con=sqlite3.connect("../input/pitchfork-data/database.sqlite")
sql="select * from artists"
df=pd.read_sql(sql,con)
df
import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
import pandas as pd
pd.DataFrame(data ={'Apples':[30],'Bananas':[21]})

import pandas as pd
pd.DataFrame(data={'Apples':[35,21],'Bananas':[21,34]}, index=['2017 Sales','2018 Sales'])
import pandas as pd
pd.Series(data=['4 cups','1 cup','2 large','1 can'], index=['Flour','Milk','Eggs','Spam'],name='Dinner')
import pandas as pd
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv')
import pandas as pd
df = pd.read_excel('../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls',sheet_name='Pregnant Women Participating')
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv('cows_and_goats')
# Your Code Here
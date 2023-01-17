import pandas as pd

import numpy as np
# Table 1

df1 = pd.DataFrame({

    'HistoryDate': ['4/5/2018', '2/4/2009', '3/5/2010', '5/6/2014', '4/13/2011'], 

    'Underlyer':['AAA', 'EEE', 'EEE','EEE', 'EEE'], 

    'MarketCap':[1400000, 2232999, 98988, 122000, 1230],

    'HistoryPrice':[34, 24.1, 20.1, 19.2, 12],

    'PctMove':[0.23, 0.03, 0.01, 0.05, 0.045]}

    )

df1.HistoryDate = pd.to_datetime(df1.HistoryDate)

df1.head()
# Table2

df2 = pd.DataFrame({

    'CADate': ['3/4/2018', '5/4/2015', '2/3/2011', '4/13/2011'], 

    'Symbol':['AAA', 'VVV', 'D', 'EEE'], 

    'CAEvent':['dividend', 'reverse merger', 'stock split', 'I am awesome']}

)

df2.CADate = pd.to_datetime(df2.CADate)

df2.head()
result = df2.merge(df1, right_on=['Underlyer', 'HistoryDate'], left_on=['Symbol', 'CADate'],how='left')

result
result = df1.merge(df2, left_on=['Underlyer'], right_on=['Symbol'],how='left')

# 在股票记录中添加一项CADate， 记录这支股票发生变化的日期。由于只join on symbol，所以只要是同样的symbol，就会有一样的CADate

result
# 选取当前日期在变动日期之后的记录

result[result.HistoryDate >= result.CADate]
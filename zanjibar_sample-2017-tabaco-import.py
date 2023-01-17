import os
import sys
import platform
from datetime import datetime as dt
import numpy as np
import pandas as pd
import sqlite3
show_tables = "select tbl_name from sqlite_master where type = 'table'"
# desc 相当　"PRAGMA table_info([テーブル名])"
desc = "PRAGMA table_info([{table}])"
import pandas.io.sql as psql
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML,Markdown

country = pd.read_csv('../input/japan-trade-statistics/country_eng.csv',
                      dtype={'Country':'str','Country_name':'str','Area':'str'})
#['japantradestatistics2', 'japan-trade-statistics']
conn = sqlite3.connect('../input/japan-trade-statistics/ym_2017.db')
sql="""
select Country,sum(Value) as Value from 
ym_2017
where exp_imp=2 and
hs6='240220'
group by Country
order by Value desc
"""[1:-1]
df = pd.read_sql(sql,conn)
# Country コードを文字列に変更
df['Country'] =  df['Country'].astype('str')
df = pd.merge(df,country,on=['Country'])
df.index = df['Country_name']
xdf = df.sort_values('Value',ascending=False).head(10)
xdf.sort_values('Value')['Value'].plot.barh()



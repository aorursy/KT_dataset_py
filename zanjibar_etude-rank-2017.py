#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sys,os
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display, HTML,Markdown
import sqlite3
import pandas.io.sql as psql

show_tables = "select tbl_name from sqlite_master where type = 'table'"
desc = "PRAGMA table_info([{table}])"
plt.style.use('ggplot') 
b_dir = '../input/japan-trade-statistics'
b2_dir = '../input/japantradestatistics2'
# attach 
conn = sqlite3.connect(b_dir + '/ym_2017.db')
attach = 'attach "../input/japantradestatistics2/trade_meta_data.db" as m'
cursor = conn.cursor()
cursor.execute(attach)
# example
['country_jpn',
 'hs2_jpn',
 'hs2_eng',
 'hs4_eng',
 'hs6_jpn',
 'hs6_eng',
 'hs9_jpn',
 'hs9_eng',
 'country_eng',
 'custom',
 'hs4_jpn']
['Country', 'Country_name', 'Area']
['hs2', 'hs2_name']
['Custom', 'custom_name']
''

# country export value rank
sql="""
select v.Country,Country_name,sum(Value) as Value
from ym_2017 v,country_eng c
where exp_imp = {exp_imp} and
v.Country = c.Country 
group by v.Country
order by Value asc
"""
exp_imp=1
df = pd.read_sql(sql.format(exp_imp=exp_imp),conn)
df.index= df['Country_name']
df.tail(10).plot.barh(y=['Value'] ,x='Country_name',alpha=0.6, figsize=(10,7))
plt.title('Country Export', size=16)
exp_imp=2
df = pd.read_sql(sql.format(exp_imp=exp_imp),conn)
df.index= df['Country_name']
df.tail(10).plot.barh(y=['Value'] ,x='Country_name',alpha=0.6, figsize=(10,7))
plt.title('Country Import', size=16)

# hs2 
sql="""
select v.hs2,hs2_name,sum(Value) as Value
from ym_2017 v,hs2_eng c
where exp_imp = {exp_imp} and
v.hs2 = c.hs2
group by v.hs2
order by Value asc
"""
exp_imp=1
df = pd.read_sql(sql.format(exp_imp=exp_imp),conn)
df.index= df['hs2']
df.tail(10).plot.barh(y=['Value'] ,x='hs2',alpha=0.6, figsize=(10,7))
plt.title('hs2 Export', size=16)
df.tail(10)[['hs2_name','Value']].sort_values('Value',ascending=False)
exp_imp=2
df = pd.read_sql(sql.format(exp_imp=exp_imp),conn)
df.index= df['hs2']
df.tail(10).plot.barh(y=['Value'] ,x='hs2',alpha=0.6, figsize=(10,7))
plt.title('hs2 Import', size=16)
df.tail(10)[['hs2_name','Value']].sort_values('Value',ascending=False)

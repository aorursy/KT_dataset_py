#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sys,os,re,json,inspect,requests,codecs,platform
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import time
import statsmodels.api as sm

from IPython.display import display, HTML,Markdown
import sqlite3
import pandas.io.sql as psql
show_tables = "select tbl_name from sqlite_master where type = 'table'"
desc = "PRAGMA table_info([{table}])"
plt.style.use('ggplot') 
#os.listdir('../input')

conn = sqlite3.connect('../input/custom_2018.db')
attach = 'attach "../input/codes.db" as code'
cursor = conn.cursor()
cursor.execute(attach)

exp_imp = 1
month   = '07'
hs6      = "870323"
lang    = "jpn"

sql="""
select v.Country,c.Country_name,sum(Value) as Value
from 
code.country_{lang} as c,custom_2018 as v
where v.exp_imp = {exp_imp} and
v.month = '{month}' and
hs6 = {hs6} and
c.Country = v.Country
group by v.Country
order by Value desc
"""
df = pd.read_sql(sql.format(month=month,exp_imp=exp_imp,hs6=hs6,lang=lang),conn)
df.head(10)
Country = '304'
sql="""
select v.Custom,c.Custom_name,sum(Value) as Value
from 
code.custom as c,custom_2018 as v
where v.exp_imp = {exp_imp} and
v.month = '{month}' and
hs6 = {hs6} and
v.Country = {Country} and
v.Custom = c.Custom
group by v.Custom
order by Value desc
"""


df = pd.read_sql(sql.format(month=month,
                            exp_imp=exp_imp,
                            Country=Country,
                            hs6=hs6,
                            lang=lang),conn)
df.head(20)
Country = '601'

df = pd.read_sql(sql.format(month=month,
                            exp_imp=exp_imp,
                            Country=Country,
                            hs6=hs6,
                            lang=lang),conn)
df.head(15)
Country = '105'

df = pd.read_sql(sql.format(month=month,
                            exp_imp=exp_imp,
                            Country=Country,
                            hs6=hs6,
                            lang=lang),conn)
df.head(10)

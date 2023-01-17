#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import sys,os,re,json,inspect,requests,codecs,platform
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML,Markdown
import sqlite3
import pandas.io.sql as psql
show_tables = "select tbl_name from sqlite_master where type = 'table'"
desc = "PRAGMA table_info([{table}])"
plt.style.use('ggplot') 
hscode='081010' # strawberry
#['japantradestatistics2/year_from_1988.db', 'japantradestatistics2/untrade-simple.db']
conn_un = sqlite3.connect('../input/japantradestatistics2/untrade-simple.db')
pd.read_sql(show_tables,conn_un)
sql ="""
select Year,exp_imp,sum(Value) as Value 
from y_all.year_1988_2016 
where hs9='{hs9}' group by exp_imp,Year
"""[1:-1]
df = pd.read_sql(sql.format(hs9=str(target_hs9)),conn)
df.index = df['Year']
sns.barplot(x='Year',y='Value',hue='exp_imp',data=df)
dir(sns)

sql = 'create table x1 as select * from xxx where exp_imp=1'
conn.execute(sql)
sql = 'create table x2 as select * from xxx where exp_imp=2'
conn.execute(sql)
sql = 'select x1.Year,x1.Value as export,x2.Value as import from x1,x2 where x1.Year=x2.Year'
df = pd.read_sql(sql,conn)
df.index = df['Year']
df.plot.bar(y=['export', 'import'], alpha=0.6, figsize=(10,3))
conn.close()
conn = sqlite3.connect(":memory:")
conn.execute("ATTACH DATABASE '" + db_ym_custom_2016 + "' AS c")
sql = """
create table xxx as select month,exp_imp,sum(Value) as Value 
from c.ym_custom_2016 where hs9="081010000" group by exp_imp,month
"""[1:-1]
conn.execute(sql)
sql = 'create table x1 as select * from xxx where exp_imp=1'
conn.execute(sql)
sql = 'create table x2 as select * from xxx where exp_imp=2'
conn.execute(sql)
sql = 'select x1.month,x1.Value as export,x2.Value as import from x1,x2 where x1.month=x2.month'
df = pd.read_sql(sql,conn)
df.index = df['month']
df.plot.bar(y=['export', 'import'], alpha=0.6, figsize=(10,3))
conn.close()
conn = sqlite3.connect(":memory:")
conn.execute("ATTACH DATABASE '" + db + "' AS y_all")
sql = """
create table xxx
as select Year,exp_imp,sum(Value) as Value 
from y_all.year_1988_2016  
where hs2 > '01' and hs2 < '25' and exp_imp=1
group by exp_imp,Year
"""[1:-1]
conn.execute(sql)
sql = 'select Year,Value from xxx'
df = pd.read_sql(sql,conn)
df.index = df['Year']
df.plot.bar(y=['Value'], alpha=0.6, figsize=(10,3))
conn.close()
conn = sqlite3.connect(":memory:")
conn.execute("ATTACH DATABASE '" + db + "' AS y_all")
sql = """
create table xxx
as select Year,exp_imp,sum(Value) as Value 
from y_all.year_1988_2016  
where hs2 > '01' and hs2 < '25' 
group by exp_imp,Year
"""[1:-1]
conn.execute(sql)
sql = 'create table x1 as select * from xxx where exp_imp=1'
conn.execute(sql)
sql = 'create table x2 as select * from xxx where exp_imp=2'
conn.execute(sql)
sql = 'select x1.Year,x1.Value as export,x2.Value as import from x1,x2 where x1.Year=x2.Year'
df = pd.read_sql(sql,conn)
df.index = df['Year']
df.plot.bar(y=['export', 'import'], alpha=0.6, figsize=(10,3))
conn.close()
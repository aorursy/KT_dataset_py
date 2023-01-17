import os,sys,re
import numpy as np
import pandas as pd
import sqlite3
import pandas.io.sql as psql
show_tables = "select tbl_name from sqlite_master where type = 'table'"
# desc 相当　"PRAGMA table_info([テーブル名])"
desc = "PRAGMA table_info([{table}])"
conn = sqlite3.connect('../input/japantradestatistics2/untrade-simple.db')

# Country ranking
sql="""
select 
origin,dest,
sum(export_val) as export,sum(import_val) as import
from hs07_2014
group by origin,dest
"""
origin_dest_export_import_df = pd.read_sql(sql,conn)
country_names_df = pd.read_sql('select * from country_names',conn)
# china (origin)
origin_dest_export_import_df[origin_dest_export_import_df['origin']=='chn'].sort_values('export',ascending=False).head()
# china (dest)
origin_dest_export_import_df[origin_dest_export_import_df['dest']=='chn'].sort_values('export',ascending=False).head()
# kor (origin) Korea heavyly depend on CHina
origin_dest_export_import_df[origin_dest_export_import_df['origin']=='kor'].sort_values('export',ascending=False).head()
# jpn (origin) jpaan exports china ,usa
origin_dest_export_import_df[origin_dest_export_import_df['origin']=='jpn'].sort_values('export',ascending=False).head(10)
import os

import sys

import re

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas.io.sql as psql

import sqlite3

from datetime import datetime as dt

import datetime

import time 

import matplotlib.pyplot as plt

%matplotlib inline





show_tables = "select tbl_name from sqlite_master where type = 'table'"

desc = "PRAGMA table_info([{table}])"



try:

    conn.close()

except:

    pass

conn = sqlite3.connect(':memory:')

cursor = conn.cursor()



#/kaggle/input/hayabusa2/hkattrp.db

#/kaggle/input/hayabusa2/aocsc.db



attach = 'attach "../input/hayabusa2/{data_name}.db" as {data_name}'



data_list = ['hkattrp','aocsc']



for data_name in data_list:

    cursor.execute(attach.format(data_name=data_name))

    

# count: aocsc 5075859 hkattrp 7501952

# desc: ['datetime', 'q1', 'q2', 'q3', 'q4', 'avx', 'avy', 'avz']

# hkattrp は、綴間違い hkattrpt が本来の綴、タイプミスでデータをつくってあります。すみません。
# conn は、global で気にせず使う

class nvdashboard():

    def __init__(self):

        self.data_list = data_list

        print(dt.now())

        

    # tgは、対象のデータ名

    # start は文字列

    # interval は秒単位

    # sample_num は、取り敢えず50 にしておく

    # 補完はなし

    # まず、epoch に変換

    # 

    

    def sampling(self,start,interval=1,sample_num=100,tg='aocsc'):

        if tg not in self.data_list:

            print('Eror...: ' + tg + ' not in data_list')

        

        start_epoch = self.to_epoch(start)

        last_epoch = start_epoch + interval*sample_num



        last = dt.strftime(self.from_epoch(last_epoch),

                           '%Y/%m/%d %H:%M:%S')+'.000'

        

        print('last: ' + str(last))

        sql = """

        select * from {tg}

        where datetime >  '{start}'

        and datetime < '{last}'

        """[1:-1]

        df = pd.read_sql(sql.format(tg=tg,start=start,last=last),conn)

        

        df['epoch'] = df['datetime']

        df['epoch'] = df['epoch'].map(self.to_epoch)

        df['s'] = df['epoch']

        df['s'] = df['s'].map(lambda x: (x-start_epoch) // interval )

        



        df_sample = df[['s', 'q1', 'q2', 'q3', 'q4', 'avx', 'avy', 'avz']].groupby('s')



        """

        print(df.shape)

        print()

        print(df.head(3))

        print(df.tail(3))

        """

        #return(df)

        return(df_sample.mean())

    

    def to_epoch(self,datetime_string):

        return(int(time.mktime(pd.to_datetime(datetime_string).timetuple())))

    

    def from_epoch(self,epoch_time):

        return(dt.fromtimestamp(epoch_time))

    



nv = nvdashboard()
# データ数のカウントですが、　0.8568644220013084 sec かかります。

sql = """

select count(*) from aocsc

"""[1:-1]

start_time = time.perf_counter()

df = pd.read_sql(sql,conn)

execution_time = time.perf_counter() - start_time

print(execution_time)
import ipywidgets as widgets

from ipywidgets import HBox,VBox

from IPython.display import display,clear_output



col_dropdown = widgets.Dropdown(

options=['q1','q2','q3','q4','avx','avy','avz'],description='Columns')



year_dropdown =widgets.Dropdown(

options=['2014','2015','2016','2017','2018'],description='year')



month_dropdown = widgets.Dropdown(

options = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11','12'] ,

description = 'month')



days = [str(x).zfill(2)  for x in  range(32)][1:]

day_dropdown = widgets.Dropdown(

options = days ,

description = 'day')



hours = [str(x).zfill(2)  for x in  range(24)]

hour_dropdown = widgets.Dropdown(

options = hours ,

description = 'hour')



minutes = [str(x).zfill(2)  for x in  range(60)]

minute_dropdown = widgets.Dropdown(

options = minutes ,

description = 'minute')



seconds = [str(x).zfill(2)  for x in  range(60)]

second_dropdown = widgets.Dropdown(

options = seconds ,

description = 'second')



intervals = [1,60,600,3600,86400]

interval_dropdown = widgets.Dropdown(

options=intervals,

description=' interval( second )')



start = '2014/12/03 04:07:05.773'

interval = 60  # 秒単位

sample_num = 10000

target_data = 'aocsc'

df = nv.sampling(start,interval,sample_num,target_data)

col_dropdown = widgets.Dropdown(

options=['q1','q2','q3','q4','avx','avy','avz'],description='Columns',value='avx')

sample_start = widgets.Dropdown(

options=[0,1000,2000,3000],description='sample_start',value=0)

sample_length = widgets.Dropdown(

options=[100,500,1000],description='sample_length')

display(col_dropdown)

display(sample_start)

display(sample_length)
df[sample_start.value:sample_start.value + sample_length.value][col_dropdown.value].plot(figsize=(20,6))
1+1
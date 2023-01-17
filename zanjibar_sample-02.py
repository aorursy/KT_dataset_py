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

class nvdashboard():

    def __init__(self):

        print(dt.now())

    # tg: table, ts: timestamp

    def sample(self,d_type,tg,ts,start,interval=1,sample_num=100):

        start_epoch = self.to_epoch(start)

        last_epoch = start_epoch + interval*sample_num

        last = dt.strftime(self.from_epoch(last_epoch),

                           '%Y/%m/%d %H:%M:%S')+'.000'

        print('last: ' + str(last))

        

        if d_type == 'sqlite':

            sql = """

            select * from {tg}

            where {ts} >  '{start}'

            and {ts} < '{last}'

            """[1:-1]

            df = pd.read_sql(sql.format(tg=tg,

            ts=ts,start=start,last=last),conn)

            df['epoch'] = df[ts]

            df['epoch'] = df['epoch'].map(self.to_epoch)

            df['s'] = df['epoch']

            df['s'] = df['s'].map(lambda x: (x-start_epoch) // interval ) # allocate quotient for interval

            df['s'] = df['s'].map(lambda x: self.from_epoch(start_epoch + x*interval)) # convert datetime from epoch

            df_sample = df[['s', 'q1', 'q2', 'q3', 'q4', 'avx', 'avy', 'avz']].groupby('s') # 

            return(df_sample.mean())

        

        if d_type == 'D5A':

            table_id = 0

            item_id = 1

            engine.open_srch(table_id,item_id)

            search_id = engine.get_srchid_list()[-1]

            ret = engine.exec_search(search_id,CONSTANT.SRCHCMD_BETWEEN,start,last)



            return(ret)

        

        print('Error')

        return()

        

    

        

    def to_epoch(self,datetime_string):

        return(int(time.mktime(pd.to_datetime(datetime_string).timetuple())))

    

    def from_epoch(self,epoch_time):

        return(dt.fromtimestamp(epoch_time))

    

    

    # Extract from list using regular expression

    def re(self,xlist,treg):

        return([x for x in xlist if re.search(treg,x)])

    def tc(self,tg):

        start_time = time.perf_counter()

        exec(tg)

        execution_time = time.perf_counter() - start_time

        print(execution_time)

        return(execution_time)

        

nv = nvdashboard()
tg = 'aocsc'

ts = 'datetime'

start = '2014/12/03 04:07:05.773'

print('start: ' + start)

interval = 60  # second

sample_num = 1000 # sampling number



start_time = time.perf_counter()

df = nv.sample('sqlite',tg,ts,start,interval,sample_num)

execution_time = time.perf_counter() - start_time

print(execution_time)
import plotly.graph_objects as go

fig = go.Figure([

                 go.Scatter(x=df.index,y=df['avx'],name='avx'),

                 go.Scatter(x=df.index,y=df['avy'],name='avy'),

                 go.Scatter(x=df.index,y=df['avz'],name='avz'),

                ])

fig.show()
from multiprocess import Pool,cpu_count

import sys

import time

import requests

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



day=time.strftime("%Y%m%d",time.gmtime())



response=requests.get("https://files.coinmarketcap.com/generated/search/quick_search.json")

json=response.json()

def proc(i,json,day,pd,time):

    url="https://coinmarketcap.com/currencies/"+json[i]["slug"]+"/historical-data/?start=20130428&end="+day

    try:

        r=pd.read_html(url)[0]

    except:

        time.sleep(10)

        r=pd.read_html(url)[0]

    r["Name"]=json[i]["name"]

    r["Symbol"]=json[i]["symbol"]

    return r



def calculate(args):

    return args[0](*args[1])



num_tasks=len(json)

pool=Pool(processes=cpu_count())

results=pool.imap(calculate,[(proc,[i,json,day,pd,time]) for i in range(num_tasks)])

la=[]

for i,r in enumerate(results):

    la.append(r)

    sys.stderr.write('\rdone {0:%}'.format((i+1)/num_tasks))

da=pd.concat(la)



da.replace("-",0,inplace=True)

da.to_csv("Put your address here",index=False,encoding="utf-8")



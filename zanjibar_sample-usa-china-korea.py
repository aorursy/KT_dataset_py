import os,sys,re,time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

from IPython.display import display, HTML



plt.style.use('ggplot') 



base_dir = "../input/"

file_list = ['hs4_eng.csv', 'country_eng.csv', 'hs9_eng.csv', 

             'ym_latest.csv', 'hs2_eng.csv', 'hs6_eng.csv']



ym_all = pd.read_csv(base_dir + "/ym_latest.csv",

                 dtype={'hs2':'str','month':'str',

                        'Country':'str','hs4':'str','hs6':'str','hs9':'str'} )



y_all = pd.read_csv(base_dir + "/year_latest.csv",

                 dtype={'hs2':'str',

                        'Country':'str','hs4':'str','hs6':'str','hs9':'str'} )



country = pd.read_csv(base_dir + "/country_eng.csv",

                     dtype={'Country':'str'})

country.index=country['Country']



hs2 = pd.read_csv(base_dir + "/hs2_eng.csv",

                     dtype={'hs2':'str'})

hs2.index=hs2['hs2']



hs4 = pd.read_csv(base_dir + "/hs4_eng.csv",

                     dtype={'hs4':'str'})

hs4.index=hs4['hs4']

hs6 = pd.read_csv(base_dir + "/hs6_eng.csv",

                     dtype={'hs6':'str'})

hs6.index=hs6['hs6']

hs9 = pd.read_csv(base_dir + "/hs9_eng.csv",

                     dtype={'hs9':'str'})

hs9.index=hs9['hs9']



x = ym_all[['Year','month']].sort_values(['Year','month']).tail(1)

last_month = x['month'].values[0]

ym_last_year = int(x['Year'].values[0])

x = y_all['Year']

start_year = x.sort_values().head(1).values[0]

last_year = x.sort_values().tail(1).values[0]
def show_period(y,title=""):

    # year or month

    try:

        y.head(1)["Value"]

        P="month"

    except:

        P="Year"

    if P=="month":

        x_y = y.groupby(["Year","month"],as_index=False)["Value"].sum()

        x_y.index= x_y[['Year',"month"]]

        x_y.plot.bar(y=['Value'] ,alpha=0.6, figsize=(12,5))

        plt.title(title, size=20)

    else:

        x_y = y.groupby(["Year"],as_index=False)["VY"].sum()

        x_y.index= x_y['Year']

        x_y.plot.bar(y=['VY'] ,alpha=0.6, figsize=(12,5))

        plt.title(title, size=20)  

         

def show_country(y,head_num=10,title=""):

    try:

        y.head(1)["Value"]

        V="Value"

    except:

        V="VY"

    x_y = y.groupby(["Country"],as_index=False )[V].sum()

    x_y = pd.merge(x_y,country,on='Country')

    x_sum = x_y[V].sum()

    x_y["percent"] = 100*(x_y[V]/x_sum)

    display(HTML(title))

    x= x_y.sort_values(V,ascending=False).head(head_num)

    display(x[[V,"Country","Country_name","percent"]])

    

    c ="x.sort_values(V).plot.barh(y=['percent'] ,x=['Country'],alpha=0.6, figsize=(12,5))"

    eval(c)



def show_hs(y,hs,head_num=10,title=""):

    try:

        y.head(1)["Value"]

        V="Value"

    except:

        V="VY"



    x_y = y.groupby([hs],as_index=False )[V].sum()

    c = "pd.merge(x_y," + hs + ",on='" + hs + "')"

    x_y = eval(c)

    

    x_sum = x_y[V].sum()

    x_y["percent"] = 100*(x_y[V]/x_sum)

    x= x_y.sort_values(V,ascending=False).head(head_num)

    display(HTML(title))





    c ="x.sort_values(V).plot.barh(y=['percent'] ,x=['" + hs + "'],alpha=0.6, figsize=(12,5))"

    display(eval(c))

    display(x[[V,hs,hs+"_name","percent"]].head(head_num))



def show_area(y,head_num=10,title=""):

    try:

        y.head(1)["Value"]

        V="Value"

    except:

        V="VY"

    x_y =y

    x_y = pd.merge(x_y,country,on="Country")

    #display(x_y.head())

    x_y = x_y.groupby("Area",as_index=False )[V].sum()

    x_sum = x_y[V].sum()

    x_y["percent"] = 100*(x_y[V]/x_sum)

    x = x_y.sort_values(V,ascending=False)

    display(HTML(title))

    display(x_y.sort_values(V,ascending=False)[[V,"Area","percent"]].head(head_num))

    c ="x.sort_values(V).plot.barh(y=['percent'] ,x=['Area'],alpha=0.6, figsize=(12,5))"

    eval(c)

    

def show_export_import(y,title=""):

    try:

        y.head(1)["Value"]

        G = '"Year","month"'

        V="Value"

    except:

        G = '"Year"'

        V="VY"

    # export

    c = "y[y['exp_imp']==1].groupby([" + G + "],as_index=False)['" + V  + "'].sum()"





    x_y_exp = eval(c)

    x_y_exp.rename(columns={V: 'export'}, inplace=True)



    c = "x_y_exp[[" + G + "]]"

    #x_y_exp.index = x_y_exp[["Year","month"]]

    x_y_exp.index = eval(c)





    c = "y[y['exp_imp']==2].groupby([" + G + "],as_index=False)['" + V  + "'].sum()"

    #print(c)

    x_y_imp = eval(c)

    x_y_imp.rename(columns={V: 'import'}, inplace=True)



        



    c = "x_y_imp[[" + G + "]]"



    x_y_imp.index = eval(c)

    

    x_exp_imp = pd.concat([x_y_exp,x_y_imp['import']],axis=1) #axis=1  add columns

    

    x_exp_imp['Year'] = x_exp_imp['Year'].astype(str)

    x_exp_imp.plot.bar(y=['export', 'import'] ,alpha=0.6, figsize=(12,5))

    plt.title(title, size=20)

    return







def hs_name(hscode):

    if len(hscode) not in [2,4,6,9]:

       return(False)

    hs = "hs" + str(len(hscode))

    c = hs + ".ix[hscode]['" + hs + "_name']"

    return(eval(c))
show_export_import(ym_all)
# 304 us

show_export_import(ym_all[ym_all['Country']=='304'])

# 105 china

display(country[country['Country']=='105']['Country_name'])

show_export_import(ym_all[ym_all['Country']=='105'])
# 103 korea

display(country[country['Country']=='103']['Country_name'])

show_export_import(ym_all[ym_all['Country']=='103'])
show_hs(y_all[ (y_all['Year']==2015) & (y_all['exp_imp']==2) & (y_all['Country']=='103')],

 'hs2')
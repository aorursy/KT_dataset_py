import pandas as pd 

import seaborn as sb

import numpy as np

import plotly

import sklearn as sk

import sqlite3

import matplotlib.pyplot as plt

import plotly.offline as offline

import plotly.graph_objs as go 



from plotly.graph_objs import *

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



init_notebook_mode(connected=True) 



%matplotlib notebook
# hde school info

hde_school = pd.read_csv("/kaggle/input/uncover/HDE/global-school-closures-covid-19.csv")



# John Hopkins data

jh_cases = pd.read_csv("/kaggle/input/uncover/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-confirmed-cases.csv")



jh_deaths = pd.read_csv("/kaggle/input/uncover/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-deaths.csv")



jh_rec = pd.read_csv("/kaggle/input/uncover/johns_hopkins_csse/2019-novel-coronavirus-covid-19-2019-ncov-data-repository-recovered.csv")
# making John Hopkins data ready for merge

jh_cases['DateTime'] = pd.to_datetime(jh_cases['date'])

jh_deaths['DateTime'] = pd.to_datetime(jh_deaths['date'])

jh_rec['DateTime'] = pd.to_datetime(jh_rec['date'])



jh_cases.rename(columns={"country_region":"country"},inplace=True)

jh_deaths.rename(columns={"country_region":"country"},inplace=True)

jh_rec.rename(columns={"country_region":"country"},inplace=True)



# joining john hopkins data

john_hop = pd.merge(jh_cases, jh_deaths,on=["country","DateTime","province_state"])

john_hop.drop(columns=["date_x","date_y","lat_x","long_x"])

jh_all = pd.merge(john_hop, jh_rec,how="left",on=["country","DateTime","province_state"])

jh_all.drop(columns=["date_x","date_y","lat_x","long_x"])



# HDE School info has country names in a column, but where different regions exist, these follow after a comma

#the following steps ensure a single consistent country field so that school info can be merged to John Hopkins data



def country(x):

    for i in jh_all["country"]:

        if i in x:

            return i



hde_school["country_1"] = hde_school["country"].apply(lambda x: country(x))



hde_school.drop_duplicates(subset=["country_1","date"],inplace=True)



hde_school.drop(columns="country",inplace=True)



hde_school.rename(columns={"country_1":"country"},inplace=True)



hde_school['DateTime'] = pd.to_datetime(hde_school['date'])
hde_school.sort_values(by=["country","DateTime"]).drop_duplicates(subset="country",keep="first",inplace=True)
# sql merge used so that school info can be merged for every date after the school closure

conn = sqlite3.connect(':memory:')



hde_school.to_sql("hde_school",conn,index=False)

jh_all.to_sql("john_hop",conn,index=False)



qry = '''

    select  

        john_hop.*,

        hde_school.scale        

    from

        john_hop left join hde_school on

        hde_school.DateTime <= john_hop.DateTime and 

        hde_school.country = john_hop.country

    '''



joint = pd.read_sql(qry,conn)
# Need to dedup by date, province and country

joint.drop_duplicates(subset=["province_state","country","DateTime"],inplace=True)



joint["scale"].fillna("None",inplace=True)

# Where no school closures - the value is set to none

def schools(x):

    if x in ["Localized","National"]:

        return 1

    else:

        return 0

    

joint["schools"] = joint["scale"].apply(lambda x: schools(x))



joint["deathRate"] = joint["deaths"] / joint["confirmed"]

# Fill in where division by 0 causes null

joint["deathRate"].fillna(0,inplace=True)

# Creating the a list of graphical data

data_slider = []



for day in joint["date_x"].unique():

    

    samp = joint[joint["date_x"] == day]

    

    for col in samp.columns:  # I transform the columns into string type so I can:

        samp[col] = samp[col].astype(str)

        

    samp["text"] = samp["country"] + "Cases: " + samp["confirmed"] + " Deaths:" + samp["deaths"] + " School Closures: " + samp["scale"] + " Recovered: " +samp["recovered"]

                    

    

    data_one_year = dict(

            type='choropleth', # type of map-plot

            colorscale = "Reds",

            reversescale = True,

            locations = samp['country'], # the column with the country

            locationmode = "country names",

            z = samp['deathRate'].astype(float)*100, # the variable I want to color-code

            text = samp['text'], # hover text

            marker = dict(     # for the lines separating states

                        line = dict (

                                  color = 'rgb(255,255,255)', 

                                  width = 2) ),               

            colorbar = dict(

                        title = "Virus Death Rate")

            ) 

       

    

    data_slider.append(data_one_year)
# Creating a slider based on the date

steps = []



for i in range(len(data_slider)):

    step = dict(method='restyle',

                args=['visible', [False] * len(data_slider)],

                label='Date {}'.format(joint["date_x"].unique()[i])) # label to be displayed for each step (year)

    step['args'][1][i] = True

    steps.append(step)

joint["date_x"].unique()



##  I create the 'sliders' object from the 'steps' 



sliders = [dict(active=0, pad={"t": 1}, steps=steps)]  

    
layout = dict(geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'),

        sliders=sliders

        )
fig = go.Figure(data=data_slider, layout=layout)



# output saved as HTML - download and explore as desired

plot(fig,validate=False)
import requests

from bs4 import BeautifulSoup

import lxml.html as lh

import pandas as pd

import re

import time

import psutil



import numpy as np

from PIL import Image

import os

from os import path

import matplotlib.pyplot as plt

import plotly.graph_objects as go



import plotly.express as px



import matplotlib as plot

from matplotlib.pyplot import figure

import seaborn as sns

sns.set(style="whitegrid")





dataset = pd.DataFrame()
URL = 'https://www.worldometers.info/coronavirus/' #the website the data is extracted

page = requests.get(URL)



soup = BeautifulSoup(page.content, 'html.parser')
table = soup.find(id='nav-tabContent')

table = table.find(id = 'nav-today')

table = table.find(id = '')

table = table.find(id = 'main_table_countries_today')
table_rows = table.find_all('tr') #finds all the tables with the tag tr in html

l = [] 

for tr in table_rows: #looping through all the table to get row data of each table

    td = tr.find_all('td') #finding all column for given tables

    row = [tr.text for tr in td]

    if len(row) == 0:

        continue  

    row = row[:8] #extraction of first 8 cloums

    l.append(row) #creating list of lists to represent the table data 

#creation of dataframe from the table of html

dataset = pd.DataFrame(l, columns=["Country","Total Cases","New Cases","Total Deaths","New Deaths","Total Recoverd","Active Cases","Serious Cases"])
def dataframeCleaner(dataset):

  

    for columnname in dataset: #looping through titles of the table 

        temp = []     

        for column in dataset[columnname]:   #geting column elements for the each title

            column = str(column)

            column = column.replace(',','')# Removing unwanted data clutter

            column = column.replace('+','')#Removing unwanted '+'sign  

            try:   #using try except block to convert datatype string to integer while avoiding error

                column = int(column)

            except:

                pass

            

            temp.append(column)

        dataset[columnname] = temp

        

    dataset = dataset.drop(dataset.tail(1).index) # Deleting the last row   

    dataset = dataset.replace(r'^\s*$', 0, regex=True)# converting empty string to 0

    return dataset
dataset
dataset = dataframeCleaner(dataset)

dataset
filename = time.strftime("%Y%m%d")

dataset.to_csv
dataset = dataset.sort_values(by ='Total Cases', ascending = 0) # sorting the rows with respect to toatal cases

num_countries = 20

plotdata = dataset.drop(dataset.tail(len(dataset["Country"]) - num_countries).index)[["Country","Total Cases"]]

plotdata
sns.set(rc={'figure.figsize':(18.7,8.27)})



cvstc = sns.barplot(x="Country", y="Total Cases", data = plotdata)
dataset = dataset.sort_values(by ='Total Deaths', ascending = 0) # sorting the rows with respect to toatal deaths

num_countries = 20

plotdata = dataset.drop(dataset.tail(len(dataset["Country"]) - num_countries).index)[["Country","Total Deaths"]]

plotdata
sns.set(rc={'figure.figsize':(18.7,8.27)})



cvstc = sns.barplot(x="Country", y="Total Deaths", data = plotdata)
dataset = dataset.sort_values(by ='Total Cases', ascending = 0) # sorting the rows with respect to toatal cases

num_countries = 20

plotdata = dataset.drop(dataset.tail(len(dataset["Country"]) - num_countries).index)[['Country','Total Deaths','Total Cases']]

plotdata
fig, ax1 = plt.subplots(figsize=(18.7,8.27))

color = 'tab:red'



ax1 = sns.barplot(x='Country', y='Total Cases', data = plotdata, palette='summer')

ax1.tick_params(axis='y')



ax2 = sns.set(style="ticks", rc={"lines.linewidth": 3.7})

ax2 = ax1.twinx()

ax2.set_ylabel('Avg Percipitation %', fontsize=16)

ax2 = sns.lineplot(x='Country', y='Total Deaths', data = plotdata, sort=False, color=color)

ax2.tick_params(axis='y', color=color)

#show plot

plt.show()
plotdata = dataset[['Country','Total Cases','Total Deaths','Total Recoverd']]

plotdata
fig, ax = plt.subplots(figsize=(8.27,8.27))

sns.regplot('Total Cases','Total Deaths', data=plotdata, ax=ax)

ax2 = ax.twinx()

sns.regplot('Total Cases','Total Recoverd', data=plotdata, ax=ax2, color='r')
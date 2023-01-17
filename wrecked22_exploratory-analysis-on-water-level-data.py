import numpy as np 

import pandas as pd

import seaborn as sns

from datetime import datetime

import matplotlib.pyplot as plt 

import os

import plotly.graph_objects as go

import plotly.express as px
#Loading Data

lvl = pd.read_csv("../input/chennai_reservoir_levels.csv")

rain = pd.read_csv("../input/chennai_reservoir_rainfall.csv")
fig = px.line(lvl, x='Date', y='POONDI')

fig.show();
fig = px.line(lvl, x='Date', y='CHOLAVARAM')

fig.show();
fig = px.line(lvl, x='Date', y='REDHILLS')

fig.show();
fig = px.line(lvl, x='Date', y='CHEMBARAMBAKKAM')

fig.show();
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))

ax1.plot(rain.REDHILLS, 'r')

ax2.plot(rain.CHOLAVARAM, 'b')

plt.show();

#Water reservoir level in million cubic feet (mcf), y axis
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

ax1.plot(rain.CHOLAVARAM)

ax2.plot(rain.POONDI,'g')

plt.show();

#Water reservoir level in million cubic feet (mcf), y axis
fig ,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.plot(rain.REDHILLS,'r')

ax2.plot(rain.CHEMBARAMBAKKAM);

#Water reservoir level in million cubic feet (mcf), y axis
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))

ax1.plot(rain.CHEMBARAMBAKKAM ,'b') 

ax2.plot(rain.POONDI )

plt.show;

#Water reservoir level in million cubic feet (mcf), y axis
corr = lvl.corr()

corr.style.background_gradient(cmap='coolwarm')
corr = rain.corr()

corr.style.background_gradient(cmap='coolwarm')
rain.Date = pd.to_datetime(rain.Date)

rain.set_index('Date', inplace=True)
rain.plot(figsize=(20,10), linewidth=3, fontsize=15)

plt.xlabel('Year', fontsize=15)

plt.ylabel('Rain Level', fontsize=15);
rain.total = rain.POONDI + rain.CHOLAVARAM + rain.REDHILLS + rain.CHEMBARAMBAKKAM

rain.total.plot(figsize=(20,10), linewidth=3, fontsize=15)

plt.xlabel('Year', fontsize=15)

plt.ylabel('Rain Level', fontsize=15);
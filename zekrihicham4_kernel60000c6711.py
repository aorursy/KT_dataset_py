import datetime

import numpy as np

import scipy as sp

import scipy.fftpack

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("/kaggle/input/datahackvize/nuitees.csv")

data.head()

df_avg = data.dropna().groupby('date').mean()

df_avg.head()

maximum = []

datemax=[]
df_avg.index.size
xplot = df_avg['dpt_09']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')

k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

df_avg.index[k]

plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_11']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]

plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_12']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]

plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_30']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]

plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_31']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_32']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_34']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_46']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_48']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_65']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_66']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_81']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['dpt_82']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



maximum.append(maxi)

datemax.append(df_avg.index[k])
xplot = df_avg['total_occitanie']



fig, ax = plt.subplots(1, 1, figsize=(10, 3))

xplot.plot(ax=ax, lw=.5)

ax.set_xlabel('Date')

ax.set_ylabel('nuitée')



k=0

maxi=np.max(xplot.values)

for i in xplot : 

    if i == maxi : 

        break 

    k=k+1

print(maxi)

df_avg.index[k]



plt.xticks(rotation=25)



#maximum.append(maxi)

#datemax.append(df_avg.index[k])
col=list(df_avg.columns.values)

col.remove('total_occitanie')

col
plt.figure(figsize=(20,4))

plt.bar(col,maximum,width = 0.5, color = 'red')

plt.xlabel('departements')

plt.ylabel('max nuitée')
moisv = np.zeros(12)

for i in range(0,13):

    moisv[int(datemax[i].split('-')[1])-1] =  moisv[int(datemax[i].split('-')[1])-1]+1

plt.figure(figsize=(20,4))

plt.bar(range(1,13),moisv,width = 0.5, color = 'red')

moisv

plt.xlabel('mois')

plt.ylabel('max')


def plotseries(dpt):

    xplot = df_avg[dpt]

    fig, ax = plt.subplots(1, 1, figsize=(20, 3))

    xplot.plot(ax=ax, lw=.5)

    ax.set_xlabel('Date')

    ax.set_ylabel('nuitée')



    

    k=0

    maxi=np.max(xplot.values)

    plt.xticks(rotation=25)

    for i in xplot : 

        if i == maxi : 

            break 

        k=k+1



    df_avg.index[k]

    return maxi,df_avg.index[k]



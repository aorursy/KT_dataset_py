import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/corona-virus-report/covid_19_clean_complete.csv")

df.head()
df['Country/Region'] = df['Country/Region'].replace({'US': 'United States',

                                                     'Taiwan*':'Taiwan'})
!pip install countryinfo
from countryinfo import CountryInfo



Country_Populations = {}

for i in sorted(list(df['Country/Region'].unique())):

    try:

        Country_Populations[i] = CountryInfo(i).population()

    except (KeyError):

        Country_Populations[i] = np.nan

        

df['Populations'] = df['Country/Region'].map(Country_Populations)

df = df[~df['Populations'].isna()] # remove countries we couldn't get the population for
df = df[df['Deaths'] >= 10]
df = df[df['Province/State'].isna()]

df['Date'] = df['Date'].astype('datetime64[ns]')
top_deaths = list((df[df['Populations'] >= 1000000].groupby(['Country/Region']).Deaths.max()\

 /df[df['Populations'] >= 1000000].groupby(['Country/Region']).Populations.max()).sort_values().index[-10:])
df.sort_values(['Country/Region','Date'], inplace=True)

df['Daily_Deaths'] = df['Deaths'].diff()



mask = df['Country/Region'] != df['Country/Region'].shift(1)

df['Daily_Deaths'][mask] = np.nan

df.head()
df['Deaths_7_Days_Rolling'] = df.groupby(['Country/Region'])['Daily_Deaths'].rolling(7).mean().values
df['%_Deaths_per_Pop'] = df['Deaths_7_Days_Rolling']/df['Populations']
df['Rank'] = df.groupby('Country/Region').Date.rank()

df.head()
filtered_df = df[(df["Country/Region"].isin(top_deaths)) & (df['Populations'] >= 1000000)]
filtered_df = filtered_df.sort_values(by='Date')

filtered_df['Days Since 10 Deaths'] =filtered_df['Rank']
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



# Initialize the FacetGrid object

pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

g = sns.FacetGrid(filtered_df, row="Country/Region", hue="Country/Region", aspect=11, height=1, palette = pal)



# Draw the densities in a few steps

g.map(sns.lineplot, 'Days Since 10 Deaths', "%_Deaths_per_Pop", clip_on=False, color="w", lw=2,estimator=None)

g.map(plt.axhline, y=0, lw=2, clip_on=False)

g.map(plt.fill_between, 'Days Since 10 Deaths','%_Deaths_per_Pop')



# Define and use a simple function to label the plot in axes coordinates

def label(x, color, label):

    ax = plt.gca()

    ax.text(0, .2, label, fontweight="bold", color=color,

            ha="left", va="center", transform=ax.transAxes)

    

g.map(label, 'Days Since 10 Deaths')



# Set the subplots to overlap

g.fig.subplots_adjust(hspace=-0.4)



# Remove axes details that don't play well with overlap

g.set_titles("")

g.set(yticks=[])

g.despine(bottom=True, left=True)
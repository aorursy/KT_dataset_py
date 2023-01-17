import pandas as pd

import numpy as np

import re

import seaborn as sns

import matplotlib.pyplot as plt
# We load the two datasets 

df_mobility = pd.read_csv('/kaggle/input/google-mobility/mobility_google.csv')

df_health = pd.read_csv('/kaggle/input/country-health-indicators/country_health_indicators_v3.csv')

df_covid = pd.read_csv('/kaggle/input/uncover/UNCOVER/worldometer/worldometer/worldometer-confirmed-cases-and-deaths-by-country-territory-or-conveyance.csv')

df_mobility= df_mobility.set_index('Country').sort_index()

df_health = df_health.set_index('Country_Region').sort_index()

df_covid = df_covid.set_index('country').sort_index()
# Select the features and merge the two datasets and 

df=pd.DataFrame()

df['Residential']=df_mobility['Residential']

df['total_deaths_per_1m_pop']=df_covid['total_deaths_per_1m_pop'] 



# change the features for proportions

health_features=['Cancers (%)','Diabetes, blood, & endocrine diseases (%)','Respiratory diseases (%)','Liver disease (%)',

                 'Diarrhea & common infectious diseases (%)','HIV/AIDS and tuberculosis (%)','Nutritional deficiencies (%)',

                'Share of deaths from smoking (%)','alcoholic_beverages']

df[health_features]=df_health[health_features]



# Parse the data from mobility database

def to_percent(x):

    try:

        return int(re.findall(r'\b\d+\b',x)[0])/100

    except :

        return np.nan

df['Residential']=df['Residential'].apply(to_percent)

df = df.dropna()
# Print the first few rows

df.head()
sns.distplot(df['Residential'])

plt.title('Median proportion of increase in Residential : '+str(np.median(df['Residential'])))

plt.show()
# Visualization of the correlations

df.corr().style.background_gradient()
g = sns.jointplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]), kind="kde").set_axis_labels("Residential", "log(total_deaths_per_1m_pop)")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
df = df[df['total_deaths_per_1m_pop']>=1]
# Visualization of the correlations

df.corr().style.background_gradient()
g = sns.jointplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]), kind="kde").set_axis_labels("Residential", "log(total_deaths_per_1m_pop)")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
f, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True, sharey=True)

cmap = sns.cubehelix_palette(dark=.2, light=.7, as_cmap=True)



for f in range(9):

    sns.scatterplot(df["Residential"], np.log(df["total_deaths_per_1m_pop"]),size=df.columns[f+2],

                    hue=df.columns[f+2], data=df,ax=axes.ravel()[f],sizes=(50, 200),palette=cmap, legend=False)

    axes.ravel()[f].set_title('plot for '+df.columns[f+2])

plt.tight_layout()
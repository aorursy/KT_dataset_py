#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
#sns.set()
sns.set(rc={'figure.figsize':(15,15)})
sns.set_palette(sns.cubehelix_palette(10, start=1.7, reverse=True))
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from ggplot import *
import os
os.listdir('../input')
loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
locations = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
theme_ids = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv')
theme_region = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')
loans.head()
loans.shape, loans.columns.values
loans.describe()
loans.describe(include=['O'])
locations.head()
locations.shape, locations.columns.values
locations.describe()
locations.describe(include=['O'])
theme_ids.head()
theme_ids.shape
theme_ids.describe()
mergetb = pd.merge(theme_ids[theme_ids.id< 700000], loans[loans.id <700000], how='outer', left_on='id', right_on='id')

mergetb.shape
mergetb.loc[mergetb['Partner ID'] != mergetb['partner_id']]
theme_region.head()
theme_region.shape
theme_region['Loan Theme ID'].nunique()
theme_region['Loan Theme Type'].nunique()
loans.info()
loans.head()
loans['borrower_genders'].value_counts()[0:10]
loans[loans.borrower_genders.isnull()].borrower_genders
def process_gender(x):
    if type(x) is float and np.isnan(x):
        return 'nan'
    genders = x.split(",")
    Male_count = [sum(g.strip()=='male' for g in genders)]
    Female_count = [sum(g.strip() == 'female' for g in genders)]
    if Female_count > Male_count:
        return "Majority Female"
    elif Female_count < Male_count:
        return "Majority Male"
    else:
        return "Equal M & F"
loans['borrower_genders'] = loans['borrower_genders'].apply(process_gender)
loans['borrower_genders'].tail()
loans.columns
sns.distplot(loans['funded_amount'])
sns.regplot(x='loan_amount', y='funded_amount', data=loans)
ggplot(loans, aes(x='loan_amount', y='funded_amount'))+geom_point()
sns.set(rc={'figure.figsize':(15,60)})
sns.countplot(y='activity', data=loans, order=loans['activity'].value_counts().index)
loans_gb_activity = loans.groupby(by='activity')
activity_rank = pd.DataFrame(loans_gb_activity.funded_amount.mean().sort_values(ascending=False).reset_index())
activity_rank.columns = ['activity', 'avg_funded_amount']
activity_rank[0:10]
sns.set(rc={'figure.figsize':(15, 60)})
sns.boxplot(x='funded_amount', y='activity', data=loans, order=activity_rank['activity'])
sns.set(rc={'figure.figsize':(15, 60)})
sns.boxplot(x='funded_amount', y='activity', data=loans[loans.funded_amount<=20000], order=activity_rank['activity'])
sns.set(rc={'figure.figsize':(8, 8)})
sns.countplot(y='sector', data=loans, order=loans['sector'].value_counts().index)
loans_gb_sector = loans.groupby(by='sector')
sector_rank = pd.DataFrame(loans_gb_sector.funded_amount.mean().sort_values(ascending=False).reset_index())
sector_rank.columns=['sector', 'avg_funded_amount']
sector_rank
sns.set(rc={'figure.figsize': (16, 16)})
sns.boxplot(x='funded_amount', y='sector', data=loans, order=sector_rank['sector'])
sns.boxplot(x='funded_amount', y='sector', data=loans[loans.funded_amount<20000], order=sector_rank['sector'])
loans_gb_sector_activity = loans.groupby(by=['sector', 'activity'])
loans_gb_sector_activity['funded_amount'].mean()
loans_gb_sector['activity'].unique(), loans_gb_sector['activity'].unique().apply(len)
sns.set(rc={'figure.figsize':(16, 80)})
fig, axs = plt.subplots(nrows=loans['sector'].nunique())
for i in range(loans['sector'].nunique()):
    sns.boxplot(x='activity', y='funded_amount', data=loans[loans.sector==loans['sector'].unique()[i]], ax=axs[i]).set_title('Funded amount distribution per activity when sector = '+loans['sector'].unique()[i])
sns.set(rc={'figure.figsize':(15,15)})
sns.countplot(y='country', data=loans, order=loans['country'].value_counts().index)
loans_gb_country = loans.groupby(by='country')
avg_funded_amount_per_country = pd.DataFrame(loans_gb_country.funded_amount.mean().sort_values(ascending=False).reset_index())
avg_funded_amount_per_country.columns=['country', 'avg_funded_amount']
sns.boxplot(x='funded_amount', y='country', data=loans, order=avg_funded_amount_per_country['country'])
loans['country'].nunique(), loans['region'].nunique()
loans_gb_country.region.unique(), loans_gb_country.region.unique().apply(len).sum()
#geo_locations = pd.read_csv('input/kiva_locations.csv')
geo_locations  = pd.read_csv('../input/kiva-challenge-coordinates/kiva_locations.csv', sep='\t', error_bad_lines=False)
geo_locations.head()
geo_locations.info()
geo_locations.groupby(['country']).region.unique().apply(len).sum()
loans = loans.merge(geo_locations, how='left', on=['country', 'region'])
locations.info()
MPI_loc = pd.merge(locations, geo_locations, how='inner', on=['country', 'region'])
MPI_loc.info()
MPI_loc = MPI_loc[MPI_loc.MPI.notnull()]
MPI_loc = MPI_loc.drop(['lat_x', 'lon'], axis=1)
MPI_loc['lat'] = MPI_loc['lat_y']
MPI_loc = MPI_loc.drop(['lat_y'], axis=1)
sns.set(rc={'figure.figsize':(8,8)})
sns.distplot(MPI_loc['MPI'])
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
scl = [[1, "rgb(255, 0, 0)"], [0.6, "rgb(0, 255,0)"]]
#[ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
#    [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]

data = [dict (
    type='scattergeo',
    lon = MPI_loc['lng'] , lat = MPI_loc['lat'] ,
    #marker=['red', 'blue'],
    text = MPI_loc['country'] +', '+MPI_loc['region'],
    mode = 'markers',
    marker = dict(
        size = 8,
        opacity = 0.5,
        reversescale = True,
        autocolorscale = False,
        #symbol = 'square',
        colorscale=scl,
        cmin = 0,
        color = MPI_loc['MPI'],
        cmax = MPI_loc['MPI'].max(),
        colorbar = dict(
            title="MPI colorscale")))]

layout = dict(
    title = 'MPI',
    geo = dict(
        scope='world',
        showland=True,
        landcolor = "rgb(225, 225, 225)",
        subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5)
    )

fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)
data = [dict (
    type='scattergeo',
    lon = geo_locations['lng'] , lat = geo_locations['lat'] ,
    #marker=['red', 'blue'],
    text = geo_locations['country'] +', '+geo_locations['region'],
    mode = 'markers',
    marker = dict(
        size = 8,
        opacity = 0.5,
        reversescale = True,
        autocolorscale = False,
        #symbol = 'square',
        #colorscale=scl,
        #cmin = 0,
        color = 'green'
        #cmax = MPI_loc['MPI'].max(),
        #colorbar = dict(
        #    title="MPI colorscale")
        ))]
layout = dict(
    title = 'Kiva borrower locations',
    geo = dict(
        scope='world',
        showland=True,
        landcolor = "rgb(225, 225, 225)"
        )
    )

fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)
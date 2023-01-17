# basic setup common to all analysis

import os
import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

%matplotlib inline
color = sns.color_palette()
py.init_notebook_mode(connected=True)

INPUT_DIR = '../input/' # location for Kaggle data files
print(os.listdir(INPUT_DIR))
KIVA_DIR = INPUT_DIR + 'data-science-for-good-kiva-crowdfunding/'
MPI_DIR = INPUT_DIR + 'mpi/'

print(os.listdir(KIVA_DIR))
# read the data - may take some time
kiva_loans = pd.read_csv(KIVA_DIR + "kiva_loans.csv")
kiva_mpi_locations = pd.read_csv(KIVA_DIR + "kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv(KIVA_DIR + "loan_theme_ids.csv")
loan_themes_by_region = pd.read_csv(KIVA_DIR + "loan_themes_by_region.csv")

# find out the shape of the data
print("kiva_loans:",kiva_loans.shape)
print("kiva_mpi_locations:",kiva_mpi_locations.shape)
print("loan_theme_ids:",loan_theme_ids.shape)
print("loan_themes_by_region",loan_themes_by_region.shape)
kiva_loans.sample(10)
kiva_mpi_locations.sample(10)
loan_theme_ids.sample(10)
loan_themes_by_region.sample(10)
def missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print ("Missing data in Loans")
missing_data(kiva_loans)
print ("Missing data in MPI Locations")
missing_data(kiva_mpi_locations)
print ("Missing data in Loan Themes")
missing_data(loan_theme_ids)
print ("Missing data in Loan Themes by Region")
missing_data(loan_themes_by_region)
loans = kiva_loans.dropna(subset = ['country','region'])
mpi = kiva_mpi_locations.dropna(subset = ['country','region'])
loan_themes_region = loan_themes_by_region.dropna(subset = ['country','region'])

# see numnber of rows dropped:
print("kiva_loans:", kiva_loans.shape)
print("after removal:", loans.shape)

print("kiva_mpi_locations:", kiva_mpi_locations.shape)
print("after removal:", mpi.shape)

print("loan_themes_by_region",loan_themes_by_region.shape)
print("after removal:", loan_themes_region.shape)
print(os.listdir(MPI_DIR))
mpi_national = pd.read_csv(MPI_DIR + "MPI_national.csv")
mpi_subnational = pd.read_csv(MPI_DIR + "MPI_subnational.csv")

# find out the shape of the data
print("mpi national:",mpi_national.shape)
print("mpi subnational:",mpi_subnational.shape)
missing_data(mpi_subnational)
# renaming some columns to make it consistent with the Kiva MPI dataset
mpi_subnational.rename(columns={'World region': 'world_region', 
                                'Sub-national region': 'region',
                                'Intensity of deprivation Regional': 'deprivation_intensity',
                                'Headcount Ratio Regional': 'headcount_ratio',
                                'Country': 'country'}, 
                       inplace=True)

mpi = pd.merge(mpi, mpi_subnational)
mpi.shape
mpi_subnational.sample(10)
plt.figure(figsize=(16,9))
sns.barplot(x=mpi.world_region.value_counts().values,
            y=mpi.world_region.value_counts().index)
plt.title("Poverty by World-Region")
plt.figure(figsize=(9,16))
sns.barplot(x=mpi.country.value_counts().values,
            y=mpi.country.value_counts().index)
plt.title("Poverty by country")
def plot_loan_purpose(df, title):
    plt.figure(figsize=(8, 8)) 
    sns.barplot(x=df.values[::-1],
                y=df.index[::-1])
    plt.title(title)

loan_sectors = loans['sector'].value_counts()[:20]
plot_loan_purpose(loan_sectors, 'Loan by Sector')
loan_activity = loans['activity'].value_counts()[:20]
plot_loan_purpose(loan_activity, 'Loan by Activity')
loan_use = loans['use'].value_counts()[:20]
plot_loan_purpose(loan_use, 'Loans by Use')
loans_mpi = pd.merge(loans, mpi, how='left')
loans_mpi.count()
df = loans_mpi.dropna(subset=['MPI'])
df.sample(10)
def reg_plot(x, y, title):
    plt.figure(figsize=(16,9))
    sns.regplot(x, y, fit_reg=True)
    plt.title(title)
    plt.show()

dlc = df.groupby(['country','region','MPI'])['loan_amount'].count().reset_index(name='loan_count')
reg_plot(dlc.MPI, dlc['loan_count'], 'MPI vs. Loan Count')
dlc.loc[dlc['loan_count'] == dlc['loan_count'].max()]
dlc = dlc.loc[dlc['loan_count'] < dlc['loan_count'].max()]
reg_plot(dlc.MPI, dlc['loan_count'], 'MPI vs. Loan Count')
dlm = df.groupby(['country','region','MPI'])['loan_amount'].median().reset_index(name='median_loan_amount')
reg_plot(dlm.MPI, dlm['median_loan_amount'], 'MPI vs. Median Loan Amount')
dlm.loc[dlm['median_loan_amount'] == dlm['median_loan_amount'].max()]
dlm = dlm.loc[dlm['median_loan_amount'] < dlm['median_loan_amount'].max()]
reg_plot(dlm.MPI, dlm['median_loan_amount'], 'MPI vs. Median Loan Amount')
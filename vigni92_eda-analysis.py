# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/startup_funding.csv")
data.head()
data.info()
data.describe(include='all')
data.describe()
# derived metrics
data['Date'] = data['Date'].astype(str)
data['Date'] = pd.to_datetime(data['Date'].str.replace('.','/').str.replace('//','/'))
data['month_year'] = data['Date'].dt.strftime('%Y-%m')
data['AmountInUSD'] = data['AmountInUSD'].str.replace(',','').astype(float)
print(data[['Date','month_year']].head())
#Total number of startups
totalStartupCount = sum(data['StartupName'].value_counts())
print("Total Number of startUps:", totalStartupCount)
# Missing Values
# Count
mvc = pd.isnull(data).sum()
# Percentage
mvp = 100 * mvc/data.shape[0]
print(mvp)

import missingno
missingno.matrix(data)
pd.isnull(data).sum()
# 82.335582% of the "Remarks" columns values are missing.
# Hence, deleting "Remarks" from table and displaying remaining data.
del data["Remarks"]
data.head()
# Outliers in funded "Amount" column
data['AmountInUSD'].plot.box(figsize=(12,5))
q3 = data['AmountInUSD'].quantile(0.75)
q1 = data['AmountInUSD'].quantile(0.25)
iqr = q3 - q1
print(iqr)

lw = q1 - 1.5*iqr
uw = q3 + 1.5*iqr
print(lw, uw)

# Those amount values lying outside the range of lw and uw are outliers.
# Frequency analysis
# City wise count of start ups
City_Ws_Cnt_p = (data['CityLocation'].value_counts()/data.shape[0]*100)
#print(City_Ws_Cnt_p)
City_Ws_Cnt_p.head(10).plot.bar(figsize=(15,8))
# Bangalore tops with more start ups
# Count of funding received in each year
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
Yearly_count = data['Year'].value_counts()
print(Yearly_count)

Yearly_count.plot.bar(figsize=(12,5))
data.groupby(['month_year']).size().plot.bar(figsize=(12,4))
iv_freq_p = data['IndustryVertical'].value_counts() / data.shape[0] * 100
iv_freq_p.head(10).plot.bar(figsize=(12,5))
# Consumer Internet vertical is prefered most by investors
sv_freq_p = data['SubVertical'].value_counts() / data.shape[0] * 100
sv_freq_p.head(10).plot.bar(figsize=(12,5))
# online pharmacy in the consumer internet vertical is prefered most by investors
in_freq_p = data['InvestorsName'].value_counts() / data.shape[0] * 100
in_freq_p.head(10).plot.bar(figsize=(12,5))
# Ratan Tata is in the 4th place in top investors
data['InvestmentType'][data['InvestmentType']=='SeedFunding'] = 'Seed Funding'
data['InvestmentType'][data['InvestmentType']=='PrivateEquity'] = 'Private Equity'
data['InvestmentType'][data['InvestmentType']=='Crowd funding'] = 'Crowd Funding'
it = data['InvestmentType'].value_counts()
print(it)
it.plot.bar(figsize=(12,5))
# Most prefered investment type by investors is Seed Funding
data["AmountInUSD"].dropna().max()

# Maximum fund to start ups
data["AmountInUSD"].dropna().min()

# Minimum fund to start ups
data["AmountInUSD"].dropna().mean()

# Average fund to start ups
data.head()
get_numeric_cols = lambda df: list(df._get_numeric_data().columns)
get_numeric_cols(data)
#print(num_cols)
def get_cat_cols(df):
    num_cols = get_numeric_cols(df)
    cat_cols = np.setdiff1d(df.columns, num_cols)
    cat_cols = np.setdiff1d(cat_cols, get_date_cols(data))
    cat_cols = np.setdiff1d(cat_cols, 'StartupName')
    return cat_cols

get_cat_cols(data)
#print(cat_cols)
def get_date_cols(df):
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    return date_cols

get_date_cols(data)
# Analysing the influence of all categorical columns on the one numerical column
# ANNOVA

get_numeric_cols = lambda df: list(df._get_numeric_data().columns)

def get_cat_cols(df):
    num_cols = get_numeric_cols(df)
    cat_cols = np.setdiff1d(df.columns, num_cols)
    return cat_cols

from scipy.stats import f_oneway

def test_1way_annova(df, cat_col, num_col):
    categories = df[cat_col].unique()
    groups = {}
    for role in categories:
        #print (role)
        subgroup = df[df[cat_col] == role][num_col].values
        groups[role] = subgroup
    #print(groups)
    stat, prob = f_oneway(*groups.values())
    return (stat, prob)
df111 = data.dropna()
df111.count()
#H0: cat_col influences num_col
#H1: cat_col does not influences num_col

df = df111
num_col = 'AmountInUSD'

from scipy.stats import f_oneway

for cat_col in get_cat_cols(df):
    stat, prob = test_1way_annova(df, cat_col, num_col)
    if prob < 0.05:
        print(prob)     # Here if p value is lesser than 0.05, Null hypothesis, H0 is accepted.
        print('%s influences %s' % (cat_col, num_col))
# Analysing the dependencies of all categorical columns on each other
# Chi-square test


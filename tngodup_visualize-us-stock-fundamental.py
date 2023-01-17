import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



companies_pd = pd.read_csv("../input/companies.csv")

indicator_pd = pd.read_csv("../input/indicators_by_company.csv")

balance_pd = pd.merge(companies_pd,indicator_pd,how='outer',on="company_id")
#combines balance sheet of the companies from 2010 - 2016

balance_pd.head()
# Create variable with TRUE if nationality is USA

asset_col = balance_pd['indicator_id'] == "Assets"

asset_pd = balance_pd[asset_col]

asset_pd.head()

#ax = sns.tsplot(time="timepoint", value="BOLD signal",

#...                 unit="subject", condition="name_latest",

#...                 data=asset_pd.head())

#ax = sns.tsplot(data=asset_pd[0:100].dropna, err_style="unit_traces")

sns.stripplot(x="name_latest", y="2011", data=asset_pd.head());
plt.scatter(asset_pd['2011'], asset_pd['2012'], c=["red", "green"])

labilities_col = balance_pd['indicator_id'] == "Liabilities"

labilities_pd = balance_pd[labilities_col]

labilities_pd.head()
# Load the example planets dataset

#planets = sns.load_dataset("planets")

#sns.lmplot("total_bill", "tip", tips, col="smoker");

#sns.stripplot(x="day", y="total_bill", data=tips);
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context="notebook", style="darkgrid", palette="Paired", font="sans- serif", font_scale=1, color_codes=True)

import warnings

import plotly.express as px

import plotly.graph_objects as go

warnings.filterwarnings("ignore")

data = pd.read_csv("../input/startup-investments-crunchbase/investments_VC.csv",encoding= 'unicode_escape')
data.head()
data.columns
data = data.drop(['permalink','homepage_url'],axis=1)

data.columns = ['name','category_list', 'market',

       'funding_total_usd', 'status', 'country_code', 'state_code', 'region',

       'city', 'funding_rounds', 'founded_at', 'founded_month',

       'founded_quarter', 'founded_year', 'first_funding_at',

       'last_funding_at', 'seed', 'venture', 'equity_crowdfunding',

       'undisclosed', 'convertible_note', 'debt_financing', 'angel', 'grant',

       'private_equity', 'post_ipo_equity', 'post_ipo_debt',

       'secondary_market', 'product_crowdfunding', 'round_A', 'round_B',

       'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']
data.info()
print(data.isnull().sum())

sns.heatmap(data.isnull(),cmap="viridis")
data.shape
data["funding_total_usd"] = data["funding_total_usd"].astype(str).apply(lambda x: x.replace(',',''))

data["funding_total_usd"] = data["funding_total_usd"].astype(str).apply(lambda x: x.replace(' ',''))

data["funding_total_usd"] = data["funding_total_usd"].astype(str).apply(lambda x: x.replace('-',''))

data['funding_total_usd'] = pd.to_numeric(data['funding_total_usd'], errors='coerce')

data["funding_total_usd"].head()
data_new = data.dropna()

fig=px.scatter_3d(data_new,x='funding_total_usd',y='market',z='region',color='status',size='funding_rounds',hover_data=['name'])

fig.show()
sns.countplot(data=data_new,y="founded_year",order=data_new["founded_year"].astype(int).value_counts()[:20].index)

sns.countplot(data = data,x="country_code",order=data["country_code"].value_counts()[:20].index)

plt.xticks(rotation=90)

plt.title("Top 20 countries for Startups")

sns.countplot(data = data,x="state_code",order=data["state_code"].value_counts()[:20].index)

plt.xticks(rotation=90)

plt.title("Top 20 States for Startups")

sns.countplot(data = data,y="region",order=data["region"].value_counts()[:20].index)

plt.title("Top 20 Region for Startups")

sns.countplot(data = data,y="city",order=data["city"].value_counts()[:20].index)

plt.title("Top 20 cities")

sns.countplot(data = data,y="market",order=data["market"].value_counts()[:20].index)

plt.title("Top 20 startups in market")
data.describe()
founded_after_2006 = data[data["founded_year"]>=2006]

sns.countplot(data=founded_after_2006,y = "founded_year",order = founded_after_2006["founded_year"].astype(int).value_counts().index)

plt.title("Startups founded after 2006")

founded_before_2006 = data[data["founded_year"]<2006]

sns.countplot(data=founded_before_2006,y = "founded_year",order = founded_before_2006["founded_year"].astype(int).value_counts()[:20].index)

plt.title("Startups founded before 2006")

sns.countplot(data = data , x = "status")
x = data["status"].value_counts()

plt.pie(x,labels=x.index,startangle=90,autopct="%1.1f%%",explode = (0,0.1,0.1))

USA_data = data[data["country_code"]=="USA"]
sns.countplot(data=USA_data,y="city",order = USA_data["city"].value_counts()[:20].index)

plt.title("Top cities in Usa")
sns.countplot(data=USA_data,y="market",order = USA_data["market"].value_counts()[:20].index)

plt.title("Top startups in usa market")
sns.countplot(data=USA_data,y="state_code",order = USA_data["state_code"].value_counts()[:20].index)

plt.title("Top states in USA")

sns.countplot(data=data[data["founded_year"]>=2000],x="founded_year")

plt.title("Startups Since 2000")

plt.xticks(rotation=90)
## Companies,who got seed fundings

data_seed = data[data["seed"]>0]
y = data_seed["status"].value_counts()

plt.pie(y,labels=y.index,startangle=180,autopct="%1.1f%%",explode = (0,0.1,0.1))

plt.title("Comapanies status who got seed funding")

## Companies who did not get seed funding

data_seed_not = data[data["seed"]==0]
z = data_seed_not["status"].value_counts()

plt.pie(z,labels=y.index,startangle=180,autopct="%1.1f%%",explode = (0,0.1,0.1))

plt.title("Comapanies status who did not get seed funding")

High_funded = data[data["funding_total_usd"]>1000000000.0]
x = High_funded["status"].value_counts()

plt.pie(x,labels=x.index,startangle=180,autopct="%1.1f%%",explode = (0,0.1,0.1))

plt.title("Comapanies status whose fundings are more than 1 Billion US$")

plt.hist(High_funded["funding_total_usd"])
sns.countplot(data=High_funded,y="country_code")
sns.countplot(data=High_funded,y="market")
sns.countplot(data=High_funded,y="founded_year")
sns.boxplot(y="funding_total_usd",data=High_funded)

data_seed_only = data[data["funding_total_usd"]==data["seed"]]
z = data_seed_only["status"].value_counts()

plt.pie(z,labels=y.index,startangle=180,autopct="%1.1f%%",explode = (0,0.1,0.1))

plt.title("Comapanies status who got only  seed funding")

plt.figure(figsize=(15,8))

sns.scatterplot(y="name",x="funding_total_usd",data=High_funded,hue="country_code",palette="Set2",size="funding_rounds")

plt.title("top 32 Highest funded companies")
plt.figure(figsize=(15,8))

sns.scatterplot(y="name",x="founded_year",data=High_funded,hue="country_code",palette="Set2",size="status")

plt.title("Founded year of top funded companies")
plt.figure(figsize=(15,8))

sns.scatterplot(y="name",x="funding_total_usd",data=High_funded,hue="country_code",style="country_code",palette="Set2",size="status")

plt.figure(figsize=(15,8))

sns.distplot(High_funded["funding_total_usd"])

plt.title("Funding distributions")



figin = go.Figure()



import plotly.express as px

import plotly.graph_objects as go



figin = go.Figure()

IndianStartup =  data[data["country_code"]=="IND"]

figin.add_trace(go.Scatter(

                x=IndianStartup['name'],

                y=IndianStartup['funding_total_usd'],

                name="",

                line_color='orange'))

figin.update_layout(title_text="funding status in india")

figin.show()
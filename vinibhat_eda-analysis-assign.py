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
data=pd.read_csv("../input/startup_funding.csv")
data.head()

data.dtypes
#data.describe(include="all")
#all the columns are of the datatype object we need to convert date column to datetime type and amount to float
# remove $ from amountusd column since it will take it as the string.
def temp(v):
    try:
        pd.to_datetime(v)
    except:
        print(v)
data["Date"].apply(lambda v: temp(v))
## data cleaning,formating and converting the amount column to float type.
data=pd.read_csv("../input/startup_funding.csv")
def temp(v):
    try:
        return pd.to_datetime(v.replace('.','/').replace('//','/'))
    except:
        print(v)
data["Date"]=data["Date"].apply(lambda v: temp(v))
data["month_year"]=data["Date"].dt.strftime("%Y-%m")
data["Year"]=data["Date"].dt.strftime("%Y")
data["amount"]=data["AmountInUSD"].str.replace(',', '').astype(float)
print(data[["Date","month_year","Year","amount"]].head())
data["amount"] = data["AmountInUSD"].str.replace(',', '').astype(float)

data.dtypes
# date is now of type datetime 
#Amount is of type float
#rest of the columns are of type object (strings) 
data.info()
data._get_numeric_data().columns
get_numeric_cols= lambda df:list(df._get_numeric_data().columns)
def get_cat_cols(df):
    num_cols=get_numeric_cols(df)
    cat_cols=np.setdiff1d(df.columns,num_cols)
    return cat_cols
get_cat_cols(data)

pd.isnull(data).sum()

# percentage of null values for all the columns.
pd.isnull(data).sum()/data.shape[0]*100
import missingno
missingno.matrix(data)

###AmountinUSD has many missing values about 35% of data is missing.
## subvertical also has many missing values
#Remarks has lot of missing values-->we can ignore/drop remarks column fron analysis

data["amount"].plot.box()
#there are lot of outliers in amountin USD column
#also anything above 98% and below 2% can be treated as outlier.
print(data["amount"].quantile(0.02))
print(data["amount"].quantile(0.98))

##Here anyting below 40000USD and anything above 100000000 USD is considered outliers
#* Apply EDA techniques to identify what influences investment amount (Column: AmountInUSD - Make sure you clean this column to convert it in to type numeric)
#** Univariate, bivariate, multivariate
yearfreq = data['Year'].value_counts().plot.bar()

    
#Year 2016 had maximum number of investments
month_year = data['month_year'].value_counts().plot.bar(figsize=(12,4))

#Month July 2016 followed by January of 2016 has large number of funding.
#univarient analysis.
#data.groupby(["month_year"]).size().plot.bar(figsize=(12,5), color="steelblue")


x=data["InvestmentType"].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(12,5), color="steelblue")

##Seed Funding and Private Equity are the most preferable type of funding 

x=data["IndustryVertical"].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(12,5), color="steelblue")

#ConsumerInternet is the Industry vertical on which highest number of investement unlike Technology
x=data["CityLocation"].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(12,5), color="steelblue")
##bangalore has highest number of investements
x=data["InvestorsName"].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(12,5), color="steelblue")

#Large number of the startup's funding are from undisclosed source
## ratan tata can be considered a special case, since all others are investment groups and he is an individual investing
x=data["SubVertical"].value_counts()/data.shape[0]*100
x.head(10).plot.bar(figsize=(12,5), color="steelblue")

##online pharmacy has highest investments

data.groupby(["month_year"])["amount"].mean().plot.bar(figsize=(12,5), color="steelblue")

# 2 months have highest average investment.. March and May of 2017 have highest investements.
#Lowest investment was seen in the month of October 2017



X=data.groupby('StartupName')['amount'].sum().sort_values(ascending=False)
X.head(10).plot.bar(figsize=(12,5), color="steelblue")

##Paytm and Flipkart are the 2 startups with highest investments put in to them

X=data.groupby('StartupName')['amount'].size().sort_values(ascending=False)
X.head(10).plot.bar(figsize=(12,5), color="steelblue")

##Swiggy is the comapany which received highest number if investments i.e, 7 investments
x=data.groupby(["IndustryVertical"])["amount"].mean().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")

## from the below graph we can see that average of people investing in online marketplace is more 
#x=data.groupby(["IndustryVertical"])["amount"].sum().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")
#x=data.groupby(["IndustryVertical"])["amount"].min().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")
#x=data.groupby(["IndustryVertical"])["amount"].max().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")
#x=data.groupby(["CityLocation"])["amount"].sum().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")
#x=data.groupby(["InvestmentType"])["amount"].sum().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")


x=data.groupby(["InvestorsName"])["amount"].sum().sort_values(ascending=False).head(10).plot.bar(figsize=(12,5), color="steelblue")
#Soft bank is the highest investor group in terms of sum invested
#get_cat_cols(data)
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2_contingency

get_numeric_cols= lambda df:list(df._get_numeric_data().columns)

def get_cat_cols(df):
    num_cols=get_numeric_cols(df)
    cat_cols=np.setdiff1d(df.columns,num_cols)
    return(cat_cols)
def df_1(df):
    
    cat_col1=get_cat_cols(df)
    
    I = [0,2,6,8,10]
    cat_col1=np.delete(cat_col1, I).tolist() # removed remarks,date,amountinusd(kept amount) columns
    
    t=[]
    t1=[]
    for i in range(len(cat_col1)):
        for j in range(i + 1, len(cat_col1)):
            
            obsv=df.groupby([cat_col1[i],cat_col1[j]]).size()
            obsv.name="Freq"
            obsv=obsv.reset_index()
            obsv=obsv.pivot_table(index=cat_col1[i],columns=cat_col1[j],values="Freq")
            stat, p, dof, exp =chi2_contingency(obsv.fillna(0).values)
            if p< 0.05:

                t1= (cat_col1[i],cat_col1[j])
              
                t.append(t1)

    return t

a=df_1(data)
#print(a)
for b in a:
    
    print( "%s is dependent on %s" %(b[0],b[1]))
data1 = data[np.isfinite(data['amount'])]
#pd.isnull(data["amount"]).sum()
from scipy.stats import f_oneway
def test_1way_annova(df,cat_col,num_col):

    categories=df[cat_col].unique()
    groups={} ### we create a empty dictionary which we will poplute dynamically
    for role in categories:
        subgroup=df[df[cat_col]==role][num_col].values ### for each role we will get the values(here monthly income) and role will be the key in dictionary
        groups[role]=subgroup
    stat,prob=f_oneway(*groups.values())
    return (stat,prob)

get_numeric_cols= lambda df:list(df._get_numeric_data().columns) ### since line to check numerical column names

def get_cat_cols(df):
    num_cols=get_numeric_cols(df)
    cat_cols=np.setdiff1d(df.columns,num_cols)
    return cat_cols

df=data1
 # to see if this categorical column influence numerical column num_col. 
num_col="amount" 

for cat_col in ['CityLocation', 'Date', 'IndustryVertical',
       'InvestmentType', 'InvestorsName', 'Remarks', 'StartupName',
       'SubVertical', 'Year']:
    stat,prob=test_1way_annova(df,cat_col,num_col)
    if prob< 0.05:
       
        print("%s influences %s" %(cat_col,num_col))


#Summary:
    
###AmountinUSD has many missing values about 35% of data is missing.
## subvertical also has many missing values
#Remarks has lot of missing values-->we can ignore/drop remarks column fron analysis
#there are a lot of outliers in amountin USD column.
    
#Year 2016 had maximum number of investments
#Month July 2016 followed by January of 2016 has large number of funding.
##Seed Funding and Private Equity are the most preferable type of funding 
#ConsumerInternet is the Industry vertical on which highest number of investement unlike Technology
##bangalore has highest number of investements
#Large number of the startup's funding are from undisclosed source
## ratan tata can be considered a special case, since all others are investment groups and he is an individual investing
##online pharmacy has highest investments
# 2 months have highest average investment.. March and May of 2017 have highest investements.
#Lowest investment was seen in the month of October 2017
##Paytm and Flipkart are the 2 startups with highest investments put in to them
##Swiggy is the comapany which received highest number if investments i.e, 7 investments
## from the graph we can see that average of people investing in online marketplace is more 
#Soft bank is the highest investor group in terms of sum invested
#Investment type and the Year column influence the amount.





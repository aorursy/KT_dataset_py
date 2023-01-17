"""Pandas is a Python package designed to do work with “labeled” and “relational” data simple and intuitive.

   Pandas is a perfect tool for data wrangling.

   It designed for quick and easy data manipulation, aggregation, and visualization.

"""

import pandas as pd

"""The most fundamental package, around which the scientific computation stack is built, is NumPy 

   (stands for Numerical Python).

   It provides an abundance of useful features for operations on n-arrays and matrices in Python.

"""

import numpy as np

"""

Python Library that is tailored for the generation of simple and powerful visualizations with ease is Matplotlib.

"""

import matplotlib.pyplot as plt

"""

Seaborn is mostly focused on the visualization of statistical models;

such visualizations include heat maps,

those that summarize the data but still depict the overall distributions.

"""

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/googleplaystore.csv")#for loding .csv data we use read_csv

data.dropna(inplace=True)#removes all the rows with atleast one null value,inplace is like conformation

"""

data of Reviews has string M in it i stands for million

we have to remove the string "M" and "," from values and 

miltiply with 1000000 if the values has 'M' in it

finaly convert to int



this method can clean the Reviewa column 

"""

def filter(per):

    if "M" in str(per) and "," in str(per):

        per = str(per).replace("M","")

        per = per.replace(",","")

        return int(per)*1000000

    elif "M" in str(per):

        per = int(str(per).replace("M",""))

        return per*1000000

    elif "," in str(per):

        per = str(per).replace(",","")

        return int(per)

    

    else:  

        return int(per)
data["Reviews"] =data["Reviews"].apply(filter) # all the values of column 'Reviews' are passed to filter method
"""

this methd is used to clean "size" column

size column contains the strings like 

'M' stands for megabyte

"Varies with device"

"k" stands for kilobyte

we convert every app size to megabytes and return as float type

"""

def filter1(per):

    per = str(per)

    if "M" in per:

        per = per.replace("M","")

        return float(per)

    elif per == "Varies with device":

        return np.NaN

    elif "k" in per:

        return float(per.replace("k",""))/1000

    else:

        return float(per)

data["Size"]=data["Size"].apply(filter1) #used to apply filter1 function 
"""

thid function is used to clean instals column

it remones i]the string "+" and ","

and returns as intiger

"""

def filter2(per):

    per = str(per)

    if "+" in per:

        per = per.replace("+","")

    if "," in per:

        per = per.replace(",","")

        

    return int(per)
data["Installs"]=data["Installs"].apply(filter2)# used to apply filter2 function"
"""

used to remove the string "$"

and convert thr price to rupies as floats

"""

def filter3(per):

    per = str(per)

    if "$" in per:

        per=per.split("$")[1]

    return (float(per)*69.44)
data["Price"]=data["Price"].apply(filter3)# used to apply filter 3 function
import plotly

print(plotly.__version__)

%matplotlib inline





import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

sns.pairplot(pd.DataFrame(list(zip(data["Rating"],data["Size"], np.log(data["Installs"]), np.log10(data["Reviews"]),data["Type"], data["Price"])), 

                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type')
sns.heatmap(data.corr(),cmap='coolwarm')
number_of_apps_in_category = data['Category'].value_counts().sort_values(ascending=True)



df = [go.Pie(labels = number_of_apps_in_category.index,values = number_of_apps_in_category.values,hoverinfo = 'label+value')]



plotly.offline.iplot(df, filename='active_category')

df = [go.Histogram(

        x = data.Rating,

        xbins = {'start': 1, 'size': 0.1, 'end' :5}

)]



print('Average app rating = ', np.mean(data['Rating']))

plotly.offline.iplot(df, filename='overall_rating_distribution')
#print('Junk apps priced above 350$')

data[['Category', 'App',"Price"]][data.Price > 200*64]
temp=pd.DataFrame(data["Content Rating"].value_counts()).reset_index()



temp.columns=['user', 'Content Rating']
plt.figure(figsize=(12,6))

sns.barplot(data=temp,x="user",y="Content Rating")
#most reviewed app rating

plt.figure(figsize=(12,6))

sns.distplot(data["Rating"],bins=10,color="red")
sns.kdeplot(data=data["Size"])

plt.title("size vs count")

plt.xlabel("")
plt.figure(figsize=(12,6))



sns.scatterplot(x = data.groupby('Category')['Rating'].mean().index, y = data.groupby('Category')['Rating'].mean().values)

plt.xlabel('Category', fontsize=13)

plt.ylabel('Rating', fontsize=13)

plt.xticks(rotation=90)

plt.title("avg rating table based on category")
most_popular_apps = data[(data["Reviews"]>10000000) ][ (data["Rating"]>=4.5)]

sns.countplot(most_popular_apps["Category"])

plt.xticks(rotation=90)
sns.set_context('talk',font_scale=1)

plt.figure(figsize=(17,13))

sns.countplot(data=data,y="Category",hue="Type")

plt.figure(figsize=(16,12))

sns.boxplot(data=data,x="Size",y="Category",palette='rainbow')
sns.countplot(x=data["Type"])
plt.figure(figsize=(17,13))

sns.countplot(data=data[data["Reviews"]>1000000],y="Category",hue="Type")

plt.title("most popular apps with 1000000+ reviews")

plt.xlabel("no of apps")
#most reviewed app rating

plt.figure(figsize=(12,6))

sns.distplot(data[data["Reviews"]>10000]["Rating"],bins=10,color="red")
sns.pairplot(pd.DataFrame(list(zip(most_popular_apps["Rating"],most_popular_apps["Size"], np.log(most_popular_apps["Installs"]), np.log10(most_popular_apps["Reviews"]),most_popular_apps["Type"], most_popular_apps["Price"])), 

                        columns=['Rating','Size', 'Installs', 'Reviews', 'Type', 'Price']), hue='Type')

plt.figure(figsize=(16,6))

sns.scatterplot(data=data[data["Reviews"]>100000],x="Size",y="Rating",hue="Type")

plt.title("apps with reviews graterthan 100000")
x=np.log(data["Installs"])

y=np.log(data["Reviews"])
popular_apps = data[(data["Installs"]>10000000) & (data["Rating"]>=4.7)]

#the most popular paid apps with decent reviews and ratings

pd.DataFrame(popular_apps[popular_apps["Type"]=="Free"][["App"]])
popular_apps = data[(data["Installs"]>100000) & (data["Rating"]>4.5)]

#the most popular paid apps with decent reviews and ratings

pd.DataFrame(popular_apps[popular_apps["Type"]=="Paid"][["App","Price"]])
mldata = data[["Reviews","Size","Installs","Price","Rating"]]

mldata.dropna(inplace=True)



X=mldata.iloc[:,0:-1].values

y = mldata.iloc[:,-1].values



from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X,y)



from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=300)
rfr.fit(xtrain,ytrain)

ypre = rfr.predict(xtest)



df=pd.DataFrame()



df["ytest"]=pd.Series(ytest)



df["ypre"] =pd.Series(ypre)

df.sample(10)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import plotly as py

import plotly.graph_objs as go

import plotly.express as px



%matplotlib inline



print("Python libraries are loaded succesfully")
# Reading the data analysts jobs csv file and display the first 20 records

df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv")

df.head(20)
df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv",index_col="Unnamed: 0",

                 na_values=[-1,-1.0,"-1","-1.0"] )

df.head(20)
# Get high level information about the dataframe columns

df.info()
df.drop(columns=["Competitors","Easy Apply"],inplace=True)

df.head(5)
df["Job Title"] = df["Job Title"].str.split(",", expand=True)[0]

df["Job Title"].value_counts()
df["Job Title"].replace(to_replace= ["Sr. Data Analyst","Sr Data Analyst",

                                     "Senior Analysts","Sr Analyst","Sr. Analyst",

                                     "Senior Contract Data Analyst","Data Analyst Senior",

                                     "Senior Analyst", "SENIOR ANALYST"],

                        value= "Senior Data Analyst",inplace=True )

df["Job Title"].replace(to_replace=  ["Jr. Data Analyst","Jr Data Analyst","Junior Analysts",

                                      "Jr Analyst","Jr. Analyst","Junior Contract Data Analyst",

                                      "Data Analyst Junior","JUNIOR ANALYST"],

                        value = "Junior Data Analyst",inplace=True)

df["Job Title"].replace(to_replace=  ["Analyst","Data analyst"],

                    value = "Data Analyst",inplace=True)

df["Job Title"].value_counts()
df.insert(2,"Min_Salary_USD_K",df["Salary Estimate"].str.split("-",expand=True)[0].str.extract('(\d+)'))

df.insert(3,"Max_Salary_USD_K",df["Salary Estimate"].str.split("-",expand=True)[1].str.extract('(\d+)'))

df.head()

df.info()
df[["Min_Salary_USD_K","Max_Salary_USD_K"]] = df[["Min_Salary_USD_K","Max_Salary_USD_K"]].fillna(0)

df[["Min_Salary_USD_K","Max_Salary_USD_K"]] = df[["Min_Salary_USD_K","Max_Salary_USD_K"]].astype(int)
df.info()
df["Company Name"] = df["Company Name"].str.split("\n", expand=True)[0]

df["Company Name"]
df.insert(8,"City",df["Location"].str.split(",", expand=True)[0].str.strip())

df.insert(9,"State",df["Location"].str.split(",", expand=True)[1].str.strip())

df.head()
df["State"].value_counts()
mapping_state = {"CA" :  "California", 

       "TX" :  "Texas",

       "NY" :  "New York",

       "IL" :  "Illinois",

       "PA" :  "Pennsylvania",

       "AZ" :  "Arizona",

       "CO" :  "Colorado",    

       "NC" :  "North California",

       "NJ" :  "New Jersey",    

       "WA" :  "Washington",

       "VA" :  "Virginia",

       "OH" :  "Ohio",

       "UT" :  "Utah",

       "FL" :  "Florida",

       "IN" :  "Indiana",

       "DE" :  "Delaware",

       "GA" :  "Georgia",

       "SC" :  "South California",    

       "KS" :  "Kansas","Arapahoe" : "Colorado"

        }

df.State = df.State.map(mapping_state)

df["State"].value_counts()
fig = px.bar(x= df["State"].value_counts().index, y= df["State"].value_counts().values, labels={"x":"State","y":'no of jobs'})

fig.update_layout(title="US state wise Data Analyst Job openings")

fig.show()
top_10_jobs = df["Job Title"].value_counts().head(10)

top_10_jobs
fig = px.bar(data_frame=top_10_jobs, x=top_10_jobs.index, y=top_10_jobs.values, title="Top 10 Job Roles",

             labels={"index":"Job Roles", "y":"No of jobs"} )

fig.show()
top_hiring_company = df["Company Name"].value_counts().head(10)

fig = px.pie(data_frame=top_hiring_company, names=top_hiring_company.index,values=top_hiring_company.values,

            labels={"index":"Company","values":"No.of Jobs"}, title="Top 10 hiring companies")

fig.show()
mean_salary = df[["State","Min_Salary_USD_K","Max_Salary_USD_K"]].groupby(by="State",

                as_index=False).mean().sort_values(by="Max_Salary_USD_K", ascending=False)

mean_salary
fig = go.Figure()

fig.add_trace(go.Bar(name ="Max Salary USD K", x=mean_salary["State"], y=mean_salary["Max_Salary_USD_K"]))

fig.add_trace(go.Bar(name ="Min Salary USD K", x=mean_salary["State"], y=mean_salary["Min_Salary_USD_K"]))

fig.update_layout(title="US State wise Max/Min Salary", yaxis_title="USD(K)")

fig.show()


cmp_size_dict ={"1 to 50 employees":1,"51 to 200 employees":2,"201 to 500 employees":3,"501 to 1000 employees":4,

         "1001 to 5000 employees":5,"5001 to 10000 employees":6,"10000+ employees":7,"Unknown":8}

company_size = df[["Size","Job Title"]].groupby(by="Size", as_index=False).count().sort_values(by="Job Title", ascending=False)

company_size["sort_company_size"] = company_size["Size"].apply(lambda cmp_size: cmp_size_dict[cmp_size])

company_size_sorted = company_size.sort_values(by="sort_company_size")

company_size_sorted
fig = px.bar(data_frame=company_size_sorted, x=company_size_sorted["Size"],y=company_size_sorted["Job Title"])

fig.update_layout(title="Job Openings VS Company Size",xaxis_title = "Company Size",yaxis_title="No of job openings")

fig.show()
sector_data = df.Sector.value_counts()

fig = px.pie(data_frame=sector_data, names=sector_data.index,values=sector_data.values,

            labels={"index":"Sector","values":"No.of Jobs"}, title="Sector wise job opportunities")

fig.show()
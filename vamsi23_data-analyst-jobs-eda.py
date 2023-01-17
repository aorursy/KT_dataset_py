import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv" , index_col = 0)

dataset.head(2)
dataset.shape
dataset.info()
dataset.describe()
dataset.describe(include = "O")
dataset.columns = dataset.columns.str.replace(" " , "_")

dataset.head(1)
dataset.Job_Title.value_counts().head()
dataset.Job_Title.replace({"Sr. Data Analyst":"Senior Data Analyst", "Sr Data Analyst":"Senior Data Analyst" , 

                           "DATA ANALYST": "Data Analyst" , "Data analyst": "Data Analyst" , 

                           "Jr Data Analyst":"Junior Data Analyst"} , inplace = True)
dataset.Salary_Estimate.value_counts().head()
sal_samp = dataset.Salary_Estimate.str.split("-" , expand = True)

dataset["Min_salary_USD_k"] = pd.to_numeric(sal_samp[0].str.extract('(\d+)' , expand = False))

dataset["Max_salary_USD_k"] = pd.to_numeric(sal_samp[1].str.extract('(\d+)' , expand = False))
dataset.loc[: , ["Min_salary_USD_k","Max_salary_USD_k"]].head(5)
dataset.Company_Name.head()
split1 = dataset.Company_Name.str.split("\n",expand = True)

dataset["Company_Name"] = split1[0]

dataset.head()
split2 = dataset.Location.str.split("," , expand = True)

dataset["City"] = split2[0]

dataset["State"] = split2[1]
dataset.State.value_counts()
dataset.loc[dataset.State.str.contains("Arapahoe") , "State"] = "CO"
dataset.State = dataset.State.str.strip()
abb = {"CA" :  "California", 

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

       "KS" :  "Kansas"  }
dataset["State"] = dataset.State.map(abb)
dataset.State.value_counts()
dataset.Size.value_counts()
dataset.Revenue.value_counts()
dataset.Industry.value_counts().head()
dataset.Easy_Apply.value_counts()
dataset.Competitors.value_counts().head()
dataset.drop(columns = "Competitors" , inplace = True)

dataset.head()
len(dataset.loc[(dataset[dataset.columns] == -1).any(1)])
len(dataset.loc[(dataset[dataset.columns] == "-1" ).any(1)])
dataset.replace(-1, np.nan , inplace = True)
dataset.replace("-1" , np.nan , inplace = True)
dataset.info()
def Missing_Values_dataframe(dataset):

        null_val = dataset.isnull().sum(0)

        

        null_val_percent = 100 * dataset.isnull().sum(0) / len(dataset)

        

        null_val_df = pd.concat([null_val, null_val_percent], axis=1)

        

        null_val_df = null_val_df.rename(mapper = {0 : "Missing Values", 1 : '% of Missing Values'} , axis = "columns")

        

        null_val_df = null_val_df.sort_values(by = "% of Missing Values" , ascending = False ).round(2)

        

        return null_val_df

Missing_Values_dataframe(dataset)
dataset["Easy_Apply"] = dataset.Easy_Apply.fillna(value = False)
dataset["Avg_salary_USD_k"] = (dataset["Max_salary_USD_k"] + dataset["Min_salary_USD_k"]) / 2
dataset["Rating_Range"] = pd.cut(dataset.Rating , bins= [0 , 2.75 , 4.2 ,5] , 

                                 labels = ["Low Rated" , "Medium Rated" , "High Rated"])
sns.catplot(kind = "box" ,x = "Rating", y = "State"  , data = dataset , height = 7 , aspect = 1.5)

plt.title("Rating of Companies VS State")
sns.relplot(kind = "scatter" , y = "State" ,x = "Min_salary_USD_k", hue = "Rating_Range",  

            s = 100,data = dataset ,height = 7 , aspect = 2 , cmap = 'viridis' )

plt.title("Minimum Salary by Companies Respect to State")
sns.relplot(kind = "scatter" , y = "State" , x = "Max_salary_USD_k", hue = "Rating_Range" ,

            data = dataset, height = 7 , aspect = 2 ,s = 100 )

plt.title(" Maximum Salary by Companies Respect to State")
state_data = dataset.groupby("State")[["Rating","Min_salary_USD_k","Max_salary_USD_k" , "Avg_salary_USD_k"]].mean()

state_data = state_data.reset_index()

state_data.sort_values("Avg_salary_USD_k" , ascending = False , inplace = True)
sns.catplot(kind = "bar" , x = "Rating" ,y = "State" , data = state_data , height = 7 , aspect = 1.5 , 

           palette = 'YlGn')

plt.xlabel("Avg_Rating")

plt.title("Avg Rating of Companies respect to States ")
sns.catplot(kind = "bar" , x = "Min_salary_USD_k" ,y = "State" , data = state_data , height = 7 , aspect = 1.5 

            , palette = 'ch:r= -0.3,l=0.95')

plt.title("Minimum Salary by State")
sns.catplot(kind = "bar" , x = "Max_salary_USD_k" ,y = "State" , data = state_data , height = 7 , aspect = 1.5 

            , palette = 'YlOrRd')

plt.title("Maximum Salary by State")
sns.catplot(kind= "bar" , x = "Avg_salary_USD_k" , y = "State" , data = state_data , height = 7 , aspect = 1.5 ,

           palette = 'PuRd')

plt.title("Average Salary by State")
dataset.info()
dataset["Salary_Range"] = pd.qcut(dataset.Avg_salary_USD_k , q = [0 , 0.4 , 0.80  , 1] ,

                                  labels = ["Low Salary" , "Medium Salary" , "High Salary"])
#dataset.loc[dataset.Salary_Range == "Low Salary"].sort_values("Avg_salary_USD_k")

#dataset.loc[dataset.Salary_Range == "Medium Salary"].sort_values("Avg_salary_USD_k")

#dataset.loc[dataset.Salary_Range == "High Salary"].sort_values("Avg_salary_USD_k")

dataset.head()
sector_data = dataset.groupby("Sector").Avg_salary_USD_k.mean()

sector_data.sort_values(ascending = False , inplace = True)

sector_data = sector_data.reset_index()

sector_data.head()
sns.catplot(kind = 'bar' , data = sector_data , y = "Sector" , x = "Avg_salary_USD_k", height = 7 , aspect = 1.5, 

           palette = "Purples")

plt.title("Avg Salary by Sector")
sector_sal_data = dataset.groupby(["Sector" , "Salary_Range"]).Avg_salary_USD_k.mean()

sector_sal_data.sort_values(ascending = False , inplace = True)

sector_sal_data = sector_sal_data.reset_index()

#pd.options.display.max_rows = 75

sector_sal_data.head(6)
sns.relplot(kind = 'scatter' , data = sector_sal_data , y = "Sector" , x = "Avg_salary_USD_k" , hue = 'Salary_Range'

            , height = 7 , aspect = 1.5  )
High_Demand_jobs = dataset.Job_Title.value_counts().head(25)

High_Demand_jobs = High_Demand_jobs.reset_index()
High_Demand_jobs.rename(columns = {"index" : "Job_Title" , "Job_Title" : "No_of_Companies"} , inplace = True)
sns.catplot(kind = 'bar' , x = "No_of_Companies" , y = "Job_Title" , data = High_Demand_jobs, height = 7 , aspect = 2)

plt.title("High_Demand_jobs")
from wordcloud import WordCloud
Job_Title=dataset['Job_Title'][~pd.isnull(dataset['Job_Title'])]

wordCloud = WordCloud(width=450,height= 300).generate(' '.join(Job_Title))

plt.figure(figsize=(15,10))

plt.axis('off')

plt.title(dataset['Job_Title'].name,fontsize=20)

plt.imshow(wordCloud)
high_paid_company_data = dataset.groupby("Company_Name").Avg_salary_USD_k.mean().sort_values(ascending = False).head(250)

high_paid_company_data.head()
comp_rata_data = dataset.loc[: , ["Company_Name" , "Rating_Range", "Job_Title","State"]]
#pd.options.display.max_rows = 350

High_Paid_jobs = comp_rata_data.merge(high_paid_company_data , how = "inner" , on = "Company_Name")

High_Paid_jobs = High_Paid_jobs.sort_values("Avg_salary_USD_k", ascending = False)

High_Paid_jobs.head(2)
High_Paid_jobs_in_high_rated_company = High_Paid_jobs.loc[High_Paid_jobs.Rating_Range == "High Rated"]

High_Paid_jobs_in_high_rated_company.head(2)
High_Paid_jobs_in_high_rated_company = High_Paid_jobs_in_high_rated_company.sort_values(by = "Avg_salary_USD_k" ,

                                                                                        ascending = False , ignore_index = True)

High_Paid_jobs_in_high_rated_company.head(2)
Top_30_High_Paid_jobs_in_high_rated_company = High_Paid_jobs_in_high_rated_company.nlargest(30 , "Avg_salary_USD_k")

Top_30_High_Paid_jobs_in_high_rated_company.head(2)
Top_23 = Top_30_High_Paid_jobs_in_high_rated_company.groupby("Job_Title").Avg_salary_USD_k.mean()

Top_23 = Top_23.reset_index()

Top_23 = Top_23.sort_values("Avg_salary_USD_k" , ascending = False , ignore_index = True)

Top_23
sns.catplot(kind = "bar" , x = "Avg_salary_USD_k"  ,y = "Job_Title" , data = Top_23 , height = 6 , aspect = 2 , 

           palette = "Reds")

plt.title("High Paid Jobs in High Rated Companies")
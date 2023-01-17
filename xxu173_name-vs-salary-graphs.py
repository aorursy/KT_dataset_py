%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv( "../input/Salaries.csv")


#pre-processing
df["EmployeeName"] = df["EmployeeName"].apply(lambda x:x.lower())
df["JobTitle"] = df["JobTitle"].apply(lambda x:x.lower())
df["namelength"] = df["EmployeeName"].apply(lambda x:len(x.split()))
df["compoundName"] = df["EmployeeName"].apply(lambda x: 1 if "-" in x else 0 )

pd.Series(df["compoundName"]).value_counts().plot(kind = "pie" , title = "compound name" , autopct='%.2f') #ccompound means has "-" 
print  (df["namelength"].value_counts())
#just for fun see the longest name:
print (df.sort_values( by = "namelength" , ascending = False)["EmployeeName"][:10])

#last name
def lastName(x):
    sp = x.split()
    if sp[-1].lower() == "jr":
        return sp[-2]
    else:
        return sp[-1]  
        
df["lastName"] = df["EmployeeName"].apply(lastName)
plt.figure()
sns.countplot(y = "lastName" , data = df , order=df["lastName"].value_counts()[:10].index)

#first name
df["firstName"] = df["EmployeeName"].apply(lambda x:x.split()[0])
plt.figure()
sns.countplot(y = "firstName" , data = df , order=df["firstName"].value_counts()[:10].index)

# exact same name
plt.figure()
sns.countplot(y = "EmployeeName" , data = df , order=df["EmployeeName"].value_counts()[:10].index)

#job titles
plt.figure()
sns.countplot(y = "JobTitle" , data = df , order=df["JobTitle"].value_counts()[:20].index)


#relation between variables
#top 3 first name by job title (top 5 job titles)
most_freq_jobs = df["JobTitle"].value_counts()[:10].index
for job in most_freq_jobs:
    print (job ,  list(  df.loc[df["JobTitle"] == job]["firstName"].value_counts()[:3].index  ) )
plt.figure()
#pre-processing the salaries: replacing NaN and strings by median
salary_types = ["BasePay" , "OvertimePay" , "OtherPay" ,"Benefits" , "TotalPay" ]
colors = ['b', 'r' , 'g' ,'k' , 'y' ]
for i,salary in enumerate(salary_types):    
    df[salary] = df[salary].fillna("Not Provided") 
    median = np.median([int(x) for x in df[salary] if not (type(x) == str)])    
    df[salary] = df[salary].apply(lambda x: median if (type(x) == str) else int(x))    
    #ax = sns.kdeplot(df[salary] , color = colors[i] )
    #ax.set_xlim(-100000,100000)

#salary distribution by jobs (take only the most freq jobs)
most_freq_jobs = df["JobTitle"].value_counts()[:10].index
for job in most_freq_jobs:    
    job_salary  = df.loc[df["JobTitle"] == job]["BasePay"]
    #overtime = df.loc[df["JobTitle"] == job]["OvertimePay"]
    plt.figure()
    ax = plt.axes()
    sns.kdeplot(job_salary , color = 'b' , ax = ax )
    #sns.kdeplot(overtime , color = 'r' , ax = ax )    
    ax.set_title(job)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("../input/who_suicide_statistics.csv")
df.head()
df.tail()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.fillna(0,inplace=True)
#after replacing with 0
df
country_wise=df["suicides_no"].groupby(df["country"]).sum().sort_values(ascending=False)
country_wise
#country_wise
country_wise_c=list(country_wise.keys())[0:20]
country_wise_c
country_wise_p=list(country_wise)[0:20]
country_wise_p
#country_vs_suicde_rate
plt.figure(figsize=(15,15))
plt.scatter(country_wise_c,country_wise_p)
plt.xlabel("country")
plt.ylabel("suicide no.")
plt.title("Top 20 countries suicide_no")
plt.xticks(rotation=45)
#gender_Wise

gender_wise=dict(df["suicides_no"].groupby(df["sex"]).sum())
gender_wise_p=list(gender_wise.values())
gender_wise_s=list(gender_wise)
#suicide_Rate according to gender
plt.bar(gender_wise_s,gender_wise_p)
plt.title("Gender wise suicide Rate")
plt.xlabel("Gender")
plt.ylabel("suicide no.")
plt.show()

#suicides year_Wise
year_wise=dict(df["suicides_no"].groupby(df["year"]).sum().sort_values(ascending=False))
year_y=list(year_wise.keys())
year_p=list(year_wise.values())
plt.figure(figsize=(10,10))
plt.bar(year_y,year_p)
plt.title("Year wise suicide Rate")
plt.xlabel("Year")
plt.ylabel("suicide no.")
plt.show()
# age _wise suicide_Rate
age_wise=dict(df["suicides_no"].groupby(df["age"]).sum())
age_wise_a=list(age_wise.keys())
age_wise_p=list(age_wise.values())
plt.figure(figsize=(10,10))
plt.bar(age_wise_a,age_wise_p)
plt.title("age wise suicide Rate")
plt.xlabel("age_group")
plt.ylabel("suicide no.")
plt.show()

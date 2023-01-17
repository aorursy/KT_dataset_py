#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# read salaries datasets
salaries = pd.read_csv('../input/Salaries.csv',low_memory=False)
#check the data
salaries.head()
#info about data
salaries.info()
#Change to object values to numeric
salaries = salaries.convert_objects(convert_numeric=True)
# now, they are numeric
salaries.info()
#median for Base Pay
round(salaries.groupby(['JobTitle'])['BasePay'].median(),3).nlargest(10)
#median for Overtime Pay
round(salaries.groupby(['JobTitle'])['OvertimePay'].median(),4).nlargest(10)
# check the tail of data
salaries.tail()
#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(salaries.isnull(),cbar=False)
#Overview of the dataset
salaries.describe(include=['O'])
#heatmap for dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(salaries.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Drop unnecessary columns
salaries = salaries.drop(["Id", "Notes", "Status", "Agency"], axis = 1)
salaries = salaries.dropna()
#Top 30 sectors for SF salary analysis
plt.figure(figsize=(13,8))
sectors = salaries['JobTitle'].value_counts()[0:30]
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.xlabel('Number of jobs', fontsize=16)
plt.ylabel("Job Title", fontsize=16)
plt.title("Number of jobs")
plt.show();
#check again the dataset
salaries.head()
#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(salaries.isnull(),cbar=False)
#calculating the mean TotalPayBenefits for each of the job titles. Print the top-10 earning job titles.
#making job title lowercase
salaries.JobTitle = salaries.JobTitle.str.lower()
#if job title has more than 500 data points as True False
criteria = salaries.JobTitle.value_counts()>500
#take job title more than 500 data points
jobtitlelist = salaries.JobTitle.value_counts()[criteria].reset_index()
#
df = salaries[['JobTitle', 'TotalPayBenefits']]
#
df = df[df.JobTitle.isin(jobtitlelist['index'])]
#
pivoted_data = df.pivot_table('TotalPayBenefits', index='JobTitle' , aggfunc=np.mean)
#
sorted_salaries = pivoted_data.sort_values(by='TotalPayBenefits', ascending= False)
print(sorted_salaries[:10])
#Visualization above the code
sorted_salaries[0:10].plot.bar()
# Let’s find the employees who earn the highest totalPay amount in each year.
salaries.JobTitle = salaries.JobTitle.str.lower()
years = salaries.Year.unique()
print(years)
for year in years:
    sorteddata = salaries[salaries.Year==year].sort_values(by='TotalPay',ascending=False)
    person = sorteddata.EmployeeName.iloc[0]
    salary = sorteddata.TotalPay.iloc[0]
    print("In the year %d %s earned the highest salary which is %8.1f"%(year,person,salary))
#Finding the number of unique job titles and print 10 most frequent job titles.
print("The number of unique job titles is %d"%len(np.unique(salaries.JobTitle)))
print("Job Title            Frequency in the dataset")
print(salaries.JobTitle.value_counts()[0:10])
#Visualization above the code
salaries.JobTitle.value_counts()[0:10].plot.bar()
#Calculating the average TotalPay values for each of the the 10 most frequent job titles
jobtitlelist = salaries.JobTitle.value_counts()[0:10]
data_10jobtitle = salaries[salaries.JobTitle.isin(jobtitlelist.index)]
avgsalary_10jobtitle=data_10jobtitle.groupby(by=data_10jobtitle.JobTitle).TotalPay.mean()
print(avgsalary_10jobtitle)
#Visualization above the code
avgsalary_10jobtitle.plot.bar()
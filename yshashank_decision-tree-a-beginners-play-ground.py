# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

data_raw=pd.read_csv("../input/HR_comma_sep.csv")

data_raw.head()

#I would like to first check for any null values in the given data set

data_raw.isnull().sum()

#Looks good, but are there any missing values?

data_raw.isnull().values.any()

#So there are no missign values. The best data seen so far.



#On a very high level let me check how are the data correlated.

data_raw.corr()



#I need to visually understand this...

import matplotlib.pyplot as plt

%matplotlib inline

plt.matshow(data_raw.corr())

plt.show()



import seaborn as sns

sns.set(color_codes=True)

#Box plot on the last evalution across job levels or categories

box=sns.boxplot(x="sales", y="last_evaluation", data=data_raw);



#The top performers seem to be distributed across Accounting//Technical//Support//Management

#The relative low performers are sales//RandD//Marketing//HR

#No specific outliers in our dataset



#How have the varios portfolios been evaluated in the recent evaluation?

port_eval=sns.barplot(x="sales", y="last_evaluation", data=data_raw);

#Looks pretty even across portfolios



#Among these portfolios, who received more promotions in the last 5 years?

last5prom=sns.barplot(x="sales", y="promotion_last_5years", data=data_raw);



#Interesting, we have "management" portfolios sweeping away the rate of promotions in the last 5 years

#IT depts have seen marginal promotions and product_mng practically nothing



#It would be interesating to see who spends more time on an avg in a month.

avg_month_hrs=sns.barplot(x="sales", y="average_montly_hours", data=data_raw);



#How many employees exist in each category?

ax = sns.countplot(x="sales", data=data_raw)



#Clearly a greater population of sales//technical//Support//IT folks as compared to:

#Product_mng//Accounting//HR//Marketing//RnD

#I would like to see now the percentage of employees under each category who left the organization



#Let me obtain all the saled folks who left the company and the ones who dint in seperate lists

df_left=data_raw[data_raw['left']==1]

df_retained=data_raw[data_raw['left']==0]

df_sales_left=df_left[df_left['sales']=="sales"]

df_sales_retained=df_retained[df_retained['sales']=="sales"]

print("Number of sales emp who left the org:", df_sales_left['left'].count())

print("Number of sales emp who still work in the org:", df_sales_retained['left'].count())

print("Percetage of sales quit::", (df_sales_left['left'].count())/(df_sales_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_sales_left['left'].count())/(df_left['left'].count())*100)

header=data_raw[:0]

    

#Accounting

df_accounting_left=df_left[df_left['sales']=="accounting"]

df_accounting_retained=df_retained[df_retained['sales']=="accounting"]



#HR

df_hr_left=df_left[df_left['sales']=="hr"]

df_hr_retained=df_retained[df_retained['sales']=="hr"]



#Technical

df_tech_left=df_left[df_left['sales']=="technical"]

df_tech_retained=df_retained[df_retained['sales']=="technical"]



#Support

df_support_left=df_left[df_left['sales']=="support"]

df_support_retained=df_retained[df_retained['sales']=="support"]



#Management

df_management_left=df_left[df_left['sales']=="management"]

df_management_retained=df_retained[df_retained['sales']=="management"]



#IT

df_it_left=df_left[df_left['sales']=="IT"]

df_it_retained=df_retained[df_retained['sales']=="IT"]



#Product_MNG

df_prod_mng_left=df_left[df_left['sales']=="product_mng"]

df_prod_mng_retained=df_retained[df_retained['sales']=="product_mng"]



#Marketing

df_marketing_left=df_left[df_left['sales']=="marketing"]

df_marketing_retained=df_retained[df_retained['sales']=="marketing"]



#RandD

df_RandD_left=df_left[df_left['sales']=="RandD"]

df_RandD_retained=df_retained[df_retained['sales']=="RandD"]



print("#################################################################################")

print("Number of accounting emp who left the org:", df_accounting_left['left'].count())

print("Number of accounting emp who still work in the org:", df_accounting_retained['left'].count())

print("Percetage of accounting  quit::", (df_accounting_left['left'].count())/(df_accounting_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_accounting_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

print("Number of hr emp who left the org:", df_hr_left['left'].count())

print("Number of hr emp who still work in the org:", df_hr_retained['left'].count())

print("Percetage of hr quit::", (df_hr_left['left'].count())/(df_hr_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_hr_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

print("Number of techie emp who left the org:", df_tech_left['left'].count())

print("Number of techie emp who still work in the org:", df_tech_retained['left'].count())

print("Percetage of techies quit::", (df_tech_left['left'].count())/(df_tech_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_tech_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

print("Number of support emp who left the org:", df_support_left['left'].count())

print("Number of support emp who still work in the org:", df_support_retained['left'].count())

print("Percetage of support quit::", (df_support_left['left'].count())/(df_support_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_support_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

print("Number of management emp who left the org:", df_management_left['left'].count())

print("Number of management emp who still work in the org:", df_management_retained['left'].count())

print("Percetage of management emp quit::", (df_management_left

['left'].count())/(df_management_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_management_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

print("Number of IT emp who left the org:", df_it_left['left'].count())

print("Number of IT emp who still work in the org:", df_it_retained['left'].count())

print("Percetage of IT emp who quit::", (df_it_left['left'].count())/(df_it_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_it_left['left'].count())/(df_left['left'].count())*100)





print("#################################################################################")

print("Number of product_mng emp who left the org:", df_prod_mng_left['left'].count())

print("Number of product_mng emp who still work in the org:", df_prod_mng_retained['left'].count())

print("Percetage of product_mng emp who quit::", (df_prod_mng_left['left'].count())/(df_prod_mng_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_prod_mng_left['left'].count())/(df_left['left'].count())*100)





print("#################################################################################")

print("Number of marketing emp who left the org:", df_marketing_left['left'].count())

print("Number of marketig emp who still work in the org:", df_marketing_retained['left'].count())

print("Percetage of marketing emp who quit::", (df_marketing_left['left'].count())/(df_marketing_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_marketing_left['left'].count())/(df_left['left'].count())*100)





print("#################################################################################")

print("Number of RnD emp who left the org:", df_RandD_left['left'].count())

print("Number of RnD emp who still work in the org:", df_RandD_retained['left'].count())

print("Percetage of RnD emp who quit::", (df_RandD_left['left'].count())/(df_RandD_retained['left'].count())*100)

print("Percenttage of the total:: ", (df_RandD_left['left'].count())/(df_left['left'].count())*100)



print("#################################################################################")

#Therefore let me order by highest attrition first:

#sales:: 28.3954074489

#techies:: 19.5183422011

#support:: 15.5418650238

#IT:: 7.64491739009

#hr:: 6.0207224867

#account:: 5.71268552226

#Marketing:: 5.68468216186

#product_mng:: 5.54466535984

#RnD:: 3.38840660879

#management:: 2.5483057967





#While the order based on promotions in the last 5 years

#Management

#Marketing

#RnD

#Sales

#HR

#Accounting

#Technical

#Support

#IT

#Product

#Hence promotions in  last 5 years will be a good independent variable to assess



#Time spent by employees who left against time spent by employees who remained

df_emp_who_left = pd.DataFrame()

df_emp_who_left = df_emp_who_left.append(data_raw)

df_emp_who_left.head()

df_timeSpent_with_category = df_emp_who_left.drop(['satisfaction_level','last_evaluation','number_project','average_montly_hours','Work_accident','promotion_last_5years','salary'],axis=1)

df_timeSpent = df_timeSpent_with_category.drop('sales',axis=1)

df_timeSpent_with_category.head()

df_timeSpent.head()



#On a high level i thouht it was important to validate the claim

sns.barplot(x="left", y="time_spend_company", data=df_timeSpent);



#Which department seems to have spent on an average more time in the company

sns.barplot(x="sales", y="time_spend_company", data=data_raw);

#Clearly:

#Management employees have a greater retention

#and Sales, techies, Support being the least



#Managent employees have a lower attrition though. So the claim on a high level may not be entirely true.

#Let me test this.

#Let me start with SLR



#Let me see the impact of time spent in the company against left, to see if this is a good variable to consider

df_timeSpent.plot(x='time_spend_company', y='left', kind='scatter')

#People with experience beyond 6 have not left the company

#Therefore i would simply infer that the most loyal resources are not leaving the company



#Based on the above analsysis i restict to the following features alone

df_reg = data_raw.drop(['number_project','average_montly_hours','salary','left','sales'],axis=1)

features = df_reg.columns

features



#Let me create the Independent, Dependent variables

X = data_raw[features]

y = data_raw['left']



#Considering 30 observations to test the model

X_test = X[0:30]

y_test = y[0:30]



#Considering the remaining for the training

X_train = X[30:]

y_train = y[30:]



from sklearn import linear_model

ols = linear_model.LinearRegression()

# Let me train the model using training data

model = ols.fit(X_train, y_train)

model.coef_

model.score(X_train, y_train)



#That was bad.. But i would not blame it on tht features. Let me try a better model to suit this case.

#Let me try a decision tree

from sklearn.tree import DecisionTreeRegressor

regr = DecisionTreeRegressor(random_state=0)

dtmodel = regr.fit(X, y)



data_raw["DecTreePrediction"]=dtmodel.predict(data_raw[features])

data_raw.head()



from sklearn import metrics

dtmodel.score(data_raw[features],data_raw['left'])

#This is a decent prediction.

#Based on the analysis above i find the major contributions from "'satisfaction level', 'time_spend_company' and promotion_last_5years" to be the key contributors to decisiding on attrition.

#And it is possible to predict if an employee is starting to think about it.
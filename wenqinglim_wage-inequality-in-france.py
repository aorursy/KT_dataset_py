import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

firms = pd.read_csv("../input/base_etablissement_par_tranche_effectif.csv")
geog = pd.read_csv("../input/name_geographic_information.csv")
salary = pd.read_csv("../input/net_salary_per_town_categories.csv")
population = pd.read_csv("../input/population.csv")
firms.head()
firms.info()
firms.describe()
firm_sizes = ['E14TS1','E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500']
firms[firm_sizes].sum().plot.bar()
firms[firm_sizes].mean().plot.bar()
for i in firm_sizes:
    new_name = i + '_P'
    total_firms = firms[firm_sizes].sum(axis=1)
    proportion = firms[i]/total_firms
    firms[new_name] = proportion

firms.head()
firms = firms.fillna(0)
firms.head()
firm_size_proportion = ['E14TS1_P','E14TS6_P', 'E14TS10_P', 'E14TS20_P', 'E14TS50_P', 'E14TS100_P', 'E14TS200_P', 'E14TS500_P']
firms[firm_size_proportion].std().plot.bar()
##TO DO: remove unimportant columns from firms table
firms_keep = ['CODGEO'] + firm_size_proportion
final_firms = firms[firms_keep]
final_firms.info()
salary.head()
salary.info()
salary.describe()
salaries_only = salary.drop(['CODGEO', 'LIBGEO'], axis=1)
salaries_only.mean().plot.bar()
salary_mf = salaries_only[['SNHMF14', 'SNHMH14']]
salary_mf.mean().plot.bar()
salary_joblevel = salaries_only[['SNHMC14', 'SNHMP14', 'SNHME14', 'SNHMO14']]
salary_joblevel_f = salaries_only[['SNHMFC14', 'SNHMFP14', 'SNHMFE14', 'SNHMFO14']]
salary_joblevel_m = salaries_only[['SNHMHC14', 'SNHMHP14', 'SNHMHE14', 'SNHMHO14']]

salary_joblevel_stack = salary_joblevel.stack().groupby(level = 1).mean().to_frame()
salary_joblevel_stack = salary_joblevel_stack.rename({'SNHMC14': 'Executives', 'SNHMP14': 'Middle Manager', 'SNHME14': 'Employee', 'SNHMO14': 'Worker'}, axis='index')
salary_joblevel_stack_f = salary_joblevel_f.stack().groupby(level = 1).mean().to_frame()
salary_joblevel_stack_f = salary_joblevel_stack_f.rename({'SNHMFC14': 'Executives', 'SNHMFP14': 'Middle Manager', 'SNHMFE14': 'Employee', 'SNHMFO14': 'Worker'}, axis='index')
salary_joblevel_stack_m = salary_joblevel_m.stack().groupby(level = 1).mean().to_frame()
salary_joblevel_stack_m = salary_joblevel_stack_m.rename({'SNHMHC14': 'Executives', 'SNHMHP14': 'Middle Manager', 'SNHMHE14': 'Employee', 'SNHMHO14': 'Worker'}, axis='index')
salary_joblevel_all = pd.concat([salary_joblevel_stack, salary_joblevel_stack_f, salary_joblevel_stack_m], axis=1)
salary_joblevel_all.columns = ['All', "Female", 'Male']
salary_joblevel_all.head()
salary_joblevel_all.plot.bar()
salary_age = salaries_only[['SNHM1814', 'SNHM2614', 'SNHM5014']]
salary_age_f = salaries_only[['SNHMF1814', 'SNHMF2614', 'SNHMF5014']]
salary_age_m = salaries_only[['SNHMH1814', 'SNHMH2614', 'SNHMH5014']]
salary_age_all = pd.concat([salary_age, salary_age_f, salary_age_m], axis=1)

salary_age_stack = salary_age.stack().groupby(level = 1).mean().to_frame()
salary_age_stack = salary_age_stack.rename({'SNHM1814': '18-25 Years Old', 'SNHM2614': '26-50 Years Old', 'SNHM5014': 'Above 50 Years Old'}, axis='index')
salary_age_stack_f = salary_age_f.stack().groupby(level = 1).mean().to_frame()
salary_age_stack_f = salary_age_stack_f.rename({'SNHMF1814': '18-25 Years Old', 'SNHMF2614': '26-50 Years Old', 'SNHMF5014': 'Above 50 Years Old'}, axis='index')
salary_age_stack_m = salary_age_m.stack().groupby(level = 1).mean().to_frame()
salary_age_stack_m = salary_age_stack_m.rename({'SNHMH1814': '18-25 Years Old', 'SNHMH2614': '26-50 Years Old', 'SNHMH5014': 'Above 50 Years Old'}, axis='index')
salary_age_all = pd.concat([salary_age_stack, salary_age_stack_f, salary_age_stack_m], axis=1)
salary_age_all.columns = ['All', "Female", 'Male']
salary_age_all.head()
salary_age_all.plot.bar()
salary_stack = salary[['CODGEO','SNHMFC14','SNHMFP14','SNHMFE14','SNHMFO14','SNHMF1814','SNHMF2614','SNHMF5014','SNHMHC14','SNHMHP14','SNHMHE14','SNHMHO14','SNHMH1814','SNHMH2614','SNHMH5014']].melt(id_vars = 'CODGEO')

def gender_label (row):
    #Labels 0 for men and 1 for women
   if row['variable'] in ['SNHMFC14','SNHMFP14','SNHMFE14','SNHMFO14','SNHMF1814','SNHMF2614','SNHMF5014'] :
      return 1
   if row['variable'] in ['SNHMHC14','SNHMHP14','SNHMHE14','SNHMHO14','SNHMH1814','SNHMH2614','SNHMH5014'] :
      return 0
   return -1

def age_label (row):
    #Labels 1 for 18-25, 2 for 26-50, 3 for above 50
   if row['variable'] in ['SNHMF1814','SNHMH1814'] :
      return 1
   if row['variable'] in ['SNHMF2614','SNHMH2614'] :
      return 2
   if row['variable'] in ['SNHMF5014','SNHMH5014'] :
      return 3 
   return 0

def job_label (row):
    #Labels 1 for worker, 2 for employee, 3 for middle manager, 4 for executive
   if row['variable'] in ['SNHMFC14','SNHMHC14'] :
      return 1
   if row['variable'] in ['SNHMFE14','SNHMHE14'] :
      return 2
   if row['variable'] in ['SNHMFP14','SNHMHP14'] :
      return 3
   if row['variable'] in ['SNHMFC14','SNHMHC14'] :
      return 4
   return 0

salary_stack['gender'] = salary_stack.apply(gender_label, axis=1)
salary_stack['age'] = salary_stack.apply(age_label, axis=1)
salary_stack['joblevel'] = salary_stack.apply(job_label, axis=1)
#print(salary_stack)
salary_stack = salary_stack.drop(columns='variable')
salary_stack.loc[(salary_stack['CODGEO'] == '01004')]
new_salary = pd.DataFrame(np.repeat(salary_stack.values,3,axis=0))
new_salary.columns = salary_stack.columns
new_salary = new_salary[new_salary.joblevel != 0]
age_labels = np.tile([1, 2, 3], int(len(new_salary.index)/3))
new_salary['age'] = age_labels

def fill_in_salary (index,row):
    # Fills in Age and new Wage in a given row
    town = row['CODGEO']
    gender = row['gender']
    joblevel = row['joblevel']
    age = row['age']
    mean1 = row['value']
    mean2 = salary_stack['value'].loc[(salary_stack['CODGEO'] == town) & (salary_stack['gender'] == gender) & (salary_stack['age'] == age)]
    mean2 = mean2.values[0]
    newmean = (mean1 + mean2)/2
    new_salary.set_value(index,'value',newmean)
      
for index, row in new_salary.iterrows():
    fill_in_salary(index,row)
    
new_salary.head(10)
new_salary.info()
new_salary.head()
#population.head()
#population.info()
#population.describe()
new_salary = new_salary.set_index('CODGEO')
final_firms = final_firms.set_index('CODGEO')
merged = pd.concat([new_salary, final_firms], join='inner', axis=1)
merged.info()
merged.head(10)
merged.info()
merged["value"] = merged.value.astype(float)
merged["gender"] = merged.gender.astype(int)
merged["joblevel"] = merged.joblevel.astype(int)
merged.reset_index()
merged.info()
merged.head()
cont_var = ['value'] + firm_size_proportion
ordinal_var = ['age', 'joblevel']
cat_var = ['gender', 'town']
cont_data = merged[cont_var]
print(cont_data.corr())
import matplotlib.pyplot as plt

pd.plotting.scatter_matrix(cont_data, figsize=(15, 15))
plt.show()
#Define target variable
merged = merged.reset_index()
merged["value"] = merged.value.astype(float)
#print(merged.info())
y = merged.value
merged = merged.drop(columns=['value', 'CODGEO'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(merged, y, test_size = 0.2, random_state = 50)
X_train.info()
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
model = linreg.fit(X_train, y_train)

# Predicting the Test set results
predictions = linreg.predict(X_test)
## Plot model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
model.score(X_test, y_test)
from sklearn.tree import DecisionTreeRegressor  
regressor = DecisionTreeRegressor()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
results=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
results.head()
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  

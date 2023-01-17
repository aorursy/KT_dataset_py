import pandas as pd
import numpy as np
import seaborn as sb
sb.set()
import matplotlib.pyplot as plt
%matplotlib inline
df1 = pd.read_csv('/kaggle/input/hr-analytics-case-study/general_data.csv')
df2 = pd.read_csv('/kaggle/input/hr-analytics-case-study/employee_survey_data.csv')
df3 = pd.read_csv('/kaggle/input/hr-analytics-case-study/manager_survey_data.csv')
df = pd.merge(df1,df2,on='EmployeeID')
df = pd.merge(df,df3,on='EmployeeID')
df.head()
df.info()
dtypes = pd.DataFrame(df.dtypes, columns=['Type'])
dtypes['Unique'] = df.nunique()
dtypes['Null'] = df.isnull().sum()
dtypes
print('Average EnvironmentSatisfaction: ',df['EnvironmentSatisfaction'].mean())
print('Average WorkLifeBalance: ',df['WorkLifeBalance'].mean())
print('Average JobSatisfaction: ',df['JobSatisfaction'].mean())
print('Average NumCompaniesWorked: ',df['NumCompaniesWorked'].mean())
print('Average TotalWorkingYears: ',df['TotalWorkingYears'].mean())
# Assigning average for null values

df['EnvironmentSatisfaction'].fillna(3, inplace=True)
df['WorkLifeBalance'].fillna(3, inplace=True)
df['JobSatisfaction'].fillna(3, inplace=True)
df['NumCompaniesWorked'].fillna(3, inplace=True)
df['TotalWorkingYears'].fillna(11, inplace=True)
plt.figure(figsize=(18,10))
sb.heatmap(df.isnull(),yticklabels=False,cmap='coolwarm')
df.drop(['EmployeeCount','Over18','StandardHours'],axis=1,inplace=True)
# Assigning Binary Values for Attrition
def convert_binary(attrition):
    if attrition == 'Yes':
        return 1
    else:
        return 0

df['Attrition'] = df['Attrition'].apply(convert_binary)
attrition_rate = df['Attrition'].value_counts()
label = ['Not Attrited', 'Attrited']

plt.pie(attrition_rate,labels=label)
plt.show()

rate_attrition = (len(df[df['Attrition'] == 1])/len(df))*100
print('Rate of Attrition for Entire Company: {:.2f}%'.format(rate_attrition))
def attrition_rate(variable,w,h):
    
    plt.figure(figsize=(w,h))
    sb.countplot(variable, hue='Attrition', data=df, palette='coolwarm')
    plt.legend(bbox_to_anchor = (1.05,1), loc=2, borderaxespad=0)
    
    df_att = df[df['Attrition']==1]
    categories = list(df[variable].unique())
    num = len(df[variable].unique())
    total_lst = []
    attrited_lst = []
    
    for var in categories:
    
        total = len(df[df[variable]==var])
        attrited = len(df_att[df_att[variable]==var])
        total_lst.append(total)
        attrited_lst.append(attrited)

    for n in range(num):
    
        percentage = (attrited_lst[n]/total_lst[n])*100
        print('Attrition Rate for {}: {:.1f}%'.format(categories[n],percentage))
def generation(age):
    if age <31:
        return 'Gen Y'
    elif age<50:
        return 'Gen X'
    else: 
        return 'Boomers'

df['Generation'] = df['Age'].apply(generation)    
attrition_rate('Generation',10,8)
attrition_rate('BusinessTravel',8,6)
attrition_rate('Department',8,5)
def location(distance):
    if distance<10:
        return 'Near'
    elif distance<20:
        return 'Moderate'
    else: 
        return 'Far'

df['Location'] = df['DistanceFromHome'].apply(location)    
plt.figure(figsize=(10,8))

df[df['Attrition']==1]['DistanceFromHome'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['DistanceFromHome'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Distance from Home')
plt.legend(legend)

attrition_rate('Location',8,6)
def education_level(level):
    if level == 1:
        return 'Below College'
    elif level == 2:
        return'College'
    elif level == 3:
        return 'Bachelor'
    elif level == 4:
        return 'Masters'
    else:
        return 'PHD'

df['Education Level'] = df['Education'].apply(education_level)    
attrition_rate('Education Level',10,5)
attrition_rate('EducationField',12,6)
gender = df['Gender'].value_counts()
label = ['Male', 'Female']

plt.pie(gender,labels=label)
plt.show()

male = (len(df[df['Gender'] == 'Male'])/len(df))*100
female = (len(df[df['Gender'] == 'Female'])/len(df))*100
print('Male Population for Entire Company: {:.0f}%'.format(male))
print('Male Population for Entire Company: {:.0f}%'.format(female))
attrition_rate('Gender',6,4)
attrition_rate('JobLevel',12,6)
attrition_rate('JobRole',22,8)
attrition_rate('MaritalStatus',8,6)
def income_tier(income):
    if income<75000:
        return 'Low Tier'
    elif income<150000:
        return 'Medium Tier'
    else: 
        return 'High Tier'

df['Income Tier'] = df['MonthlyIncome'].apply(income_tier)    
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['MonthlyIncome'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['MonthlyIncome'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Income')
plt.legend(legend)

attrition_rate('Income Tier',8,6)
attrition_rate('NumCompaniesWorked',20,10)
df['PercentSalaryHike'].value_counts()
def hike_tier(percent):
    if percent<15:
        return 'Low Hike'
    elif percent<20:
        return 'Medium Hike'
    else: 
        return 'High Hike'

df['Salary Hike %'] = df['PercentSalaryHike'].apply(hike_tier)    
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['PercentSalaryHike'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['PercentSalaryHike'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Hike Percentage')
plt.legend(legend)

attrition_rate('Salary Hike %',8,6)
attrition_rate('StockOptionLevel',8,6)
def working_years(years):
    if years<10:
        return 'Low Experience'
    elif years<25:
        return 'Moderate Experience'
    else: 
        return 'High Experience'

df['Experience'] = df['TotalWorkingYears'].apply(working_years)    
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['TotalWorkingYears'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['TotalWorkingYears'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Working Years')
plt.legend(legend)

attrition_rate('Experience',8,6)
attrition_rate('TrainingTimesLastYear',18,10)
def seniority(year):
    if year<8:
        return 'Junior Employee'
    elif year<25:
        return 'Senior Employee'
    else: 
        return 'Veteran'

df['Seniority'] = df['YearsAtCompany'].apply(seniority)
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['YearsAtCompany'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['YearsAtCompany'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Years with Company')
plt.legend(legend)

attrition_rate('Seniority',8,6)
def last_promoted(year):
    if year<4:
        return 'Recent'
    elif year<10:
        return 'Not Very Recent'
    else: 
        return 'Stagnant'

df['Last Promotion'] = df['YearsSinceLastPromotion'].apply(last_promoted)    
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['YearsSinceLastPromotion'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['YearsSinceLastPromotion'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Years Since Last Promotion')
plt.legend(legend)

attrition_rate('Last Promotion',8,6)
def with_manager(year):
    if year<5:
        return 'Short Term'
    elif year<10:
        return 'Long Term'
    else: 
        return 'Life Long'

df['With Current Manager'] = df['YearsWithCurrManager'].apply(with_manager)    
plt.figure(figsize=(10,8))
df[df['Attrition']==1]['YearsWithCurrManager'].hist(color='blue',alpha=1)
df[df['Attrition']==0]['YearsWithCurrManager'].hist(color='red',alpha=0.3)
legend = ['Attrited','Not Attrited']
plt.xlabel('Years with Current Manager')
plt.legend(legend)

attrition_rate('With Current Manager',8,6)
attrition_rate('EnvironmentSatisfaction',8,6)
attrition_rate('JobInvolvement',8,6)
attrition_rate('JobSatisfaction',8,6)
attrition_rate('WorkLifeBalance',8,6)
attrition_rate('PerformanceRating',8,6)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
logmodel = LogisticRegression()
scaler = StandardScaler()
df_selected = df.drop(['Generation','Location', 'Education Level', 
                       'Income Tier', 'Salary Hike %','Experience', 
                       'Seniority', 'Last Promotion', 'With Current Manager'
                      ], axis=1)
df_contfeats = df_selected.drop(cat_feats,axis=1)
df_contfeats = df_contfeats.drop('Attrition',axis=1)
scaler.fit(df_contfeats)
df_contfeats_scaled = scaler.transform(df_contfeats)
df_contfeats_scaled = pd.DataFrame(df_contfeats_scaled)
cat_feats = ['BusinessTravel','Department','EducationField','Gender', 
             'JobRole','MaritalStatus']
df_catfeats = df_selected[cat_feats]
df_catfeats_converted = pd.get_dummies(df_catfeats,drop_first=True)
model = pd.concat([df_contfeats_scaled,df_catfeats_converted],axis=1)
model = pd.concat([df[['Attrition']],model],axis=1)
model.head()
X = model.drop('Attrition',axis=1)
y = model['Attrition']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
print('')
print('Classification Accuracy: {:.3f}'.format(logmodel.score(X_test,y_test)))
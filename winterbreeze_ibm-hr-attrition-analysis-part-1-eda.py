from PIL import Image

Image.open('/kaggle/input/attritionpic/attrition.png')
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows',None)

pd.set_option('display.max_column',None)
data=pd.read_csv('/kaggle/input/ibmhrattritionanalysis/IBM-HR-Employee-Attrition.csv',index_col='EmployeeNumber')

data.head()                              # Made EmployeeNumber as the index 
print('The total number of rows:{} and columns:{}'.format(data.shape[0],data.shape[1]))
data.info()
data.describe()                   # Looking into Numerical Features 
data.describe(include='object')  # Looking into the Categorical Features
data[['DailyRate','HourlyRate','MonthlyRate','StandardHours']].describe()
data['HourlyRate'].plot(kind='kde') 

print('Skewness for Hourly Rate is :' ,data['HourlyRate'].skew())

print('Kurtosis for Hourly Rate is :' ,data['HourlyRate'].kurt())
data['MonthlyRate'].plot(kind='kde') 

print('Skewness for Hourly Rate is :' ,data['MonthlyRate'].skew())

print('Kurtosis for Hourly Rate is :' ,data['MonthlyRate'].kurt())
data['DailyRate'].plot(kind='kde') 

print('Skewness for Hourly Rate is :' ,data['DailyRate'].skew())

print('Kurtosis for Hourly Rate is :' ,data['DailyRate'].kurt())
pd.DataFrame({'Count':data.isnull().sum(),'Missing%':data.isnull().mean()*100}).T # No missing values 
data.drop(['EmployeeCount','Over18','StandardHours'],axis=1,inplace=True) # Insignificant Features 
data.hist(figsize=(16,12))

plt.tight_layout()
for i in data.select_dtypes(exclude='O'):

    if data[i].skew() > 0.9:

        print(i,':',data[i].skew())             #These are the numerical columns with high skewness 
for i in data.select_dtypes(exclude='O'):

    if data[i].skew() > 0.9:

        data[i]=data[i].transform(lambda x:np.log1p(x))
for i in data.select_dtypes(exclude='O'):

    if data[i].skew() > 0:

        print(i,':',data[i].skew())

        
# Checking if there is any relation between columns 
data['Education'].unique()
data['EducationField'].unique()
data['JobLevel'].unique()
data['JobRole'].unique()
'''for i in data.select_dtypes(include='O'):

    for j in data.select_dtypes(include='int64'):

        plt.subplots()

        sns.boxplot(x=i,y=j,hue='Attrition',data=data)'''   
plt.figure(figsize=(20,8))

sns.set(style='whitegrid')

data['Attrition'].value_counts().plot(kind='pie',explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=['c','r'])

print(data['Attrition'].value_counts())
print(round(pd.crosstab(data['Attrition'],data['Gender'],normalize=True)*100,2))



plt.figure(figsize=(22,16))

sns.set(style="darkgrid")

plt.subplot(221)

plt.title('Gender Vs Attrition')

sns.countplot('Gender',hue='Attrition',data=data,palette='seismic_r')



plt.subplot(222)

plt.title('Gender Vs MonthlyIncome')

sns.boxplot(data['Gender'],data['MonthlyIncome'])
pd.pivot_table(data=data,index=['Gender'],values=['MonthlyIncome'],aggfunc='mean').style.set_properties(**{'background-color': 'black',

                                                                                                               'color': 'lawngreen',

                                                                                                               })
print (pd.crosstab(data['Attrition'],data['Department']))



plt.figure(figsize=(22,16))

sns.set(style="darkgrid")

plt.subplot(221)

plt.title('Department Vs Attrition')

sns.countplot(data['Department'],hue=data['Attrition'],palette='prism_r')



plt.subplot(222)

plt.title('Department Vs MonthlyIncome')

sns.boxplot(data['Department'],data['MonthlyIncome'])
#Average salary b/w different departments 

pd.pivot_table(data=data,index=['Department'],values=['MonthlyIncome'],aggfunc='mean').style.set_properties(**{'background-color': 'black',

                                                                                                               'color': 'lawngreen',

                                                                                                               })
pd.crosstab([data['Attrition']],data['JobRole'],normalize=True)*100
plt.figure(figsize=(25,14))

sns.set(style="whitegrid")





plt.subplot(211)

plt.title('JobRole Vs Attrition')

sns.countplot(data['JobRole'],palette='afmhot_r',hue=data['Attrition'].sort_values(ascending=True))

plt.title('Attrition amongst different JobRoles',size=15)





plt.subplot(212)

plt.title('JobRole Vs MonthlyIncome')

sns.boxplot(data['JobRole'],data['MonthlyIncome'])
pd.pivot_table(data=data,index=['JobRole'],values=['MonthlyIncome'],aggfunc='mean').sort_values(by='MonthlyIncome').style.set_properties(**{'background-color': 'black',

                                                                                                               'color': 'lawngreen',

                                                                                                               })
print(pd.crosstab(data['JobInvolvement'],data['Attrition']))



plt.figure(figsize=(22,16))

plt.subplot(221)

plt.title('JobInvolvement Vs Attrition')

sns.countplot(data['JobInvolvement'],hue=data['Attrition'],palette='rocket')



plt.subplot(222)

plt.title('JobInvolvement Vs Monthly Income')

sns.boxplot(data['JobInvolvement'],data['MonthlyIncome'])
plt.figure(figsize=(22,16))

plt.subplot(221)

plt.title('Education Vs Attrition')

sns.countplot(data['Education'],hue=data['Attrition'],palette='gnuplot2')





plt.subplot(222)

plt.title('Education Vs MonthlyIncome')

sns.boxplot(data['Education'],data['MonthlyIncome'])
#Average salary b/w Education levels

pd.pivot_table(data=data,index=['Education'],values=['MonthlyIncome'],aggfunc='mean').style.set_properties(**{'background-color': 'black',

                                                                                                               'color': 'lawngreen',

                                                                                                              })  
print(pd.crosstab(columns=data['Attrition'],index=data['EducationField']))



plt.figure(figsize=(22,16))

plt.subplot(221)

plt.title('EducationField Vs Attriton')

sns.countplot(data['EducationField'],hue=data['Attrition'],palette='RdBu')



plt.subplot(222)

plt.title('EducationField Vs Monthly Income')

sns.boxplot(data['EducationField'],data['MonthlyIncome'])
agebins=pd.cut(data['Age'],bins=[15,20,25,30,35,40,45,50,55,60]) #Discretisation to understand what age categories to Target
plt.figure(figsize=(15,5))

plt.title('Distribution of Age',size=15)

sns.distplot(data['Age'],bins=[15,20,25,30,35,40,45,50,55,60],color='c')



plt.figure(figsize=(15,5))

plt.title('Age Wise Binning wrt Attrition',size=15)

sns.countplot(agebins,hue='Attrition',data=data,palette='CMRmap_r')
print(pd.crosstab(data['Attrition'],data['EnvironmentSatisfaction']))



plt.figure(figsize=(22,16))

plt.subplot(221)

plt.title('EnvironmentSatisfaction Vs MonthlyIncome')

sns.countplot(data['EnvironmentSatisfaction'],hue=data['Attrition'],palette='mako')





plt.subplot(222)

plt.title('EnvironmentSatisfaction Vs MonthlyIncome')

sns.boxplot(data['EnvironmentSatisfaction'],data['MonthlyIncome'])
data['EnvironmentSatisfaction'].value_counts().sort_values()
print(pd.crosstab(data['Attrition'],data['MaritalStatus']))



plt.figure(figsize=(20,14))

plt.subplot(221)

plt.title('MaritalStatus Vs MonthlyIncome')

sns.countplot(data['MaritalStatus'],hue=data['Attrition'],palette='hot')



plt.subplot(222)

plt.title('MaritalStatus Vs MonthlyIncome')

sns.boxplot(data['MaritalStatus'],data['MonthlyIncome'])
plt.figure(figsize=(20,12))

plt.subplot(211)

plt.title('YearsInCurrentRole Vs MonthlyIncome')

sns.countplot(data['YearsInCurrentRole'],hue=data['Attrition'],palette='mako')





plt.subplot(212)

plt.title('YearsInCurrentRole Vs MonthlyIncome')

sns.boxplot(data['YearsInCurrentRole'],data['MonthlyIncome'])
len(data.loc[(data['YearsInCurrentRole']==7) | (data['YearsInCurrentRole']==8) | (data['YearsInCurrentRole']==9) & (data['Attrition']=='Yes')])/data.shape[0]*100
plt.figure(figsize=(20,12))

plt.subplot(211)

plt.title('YearsAtCompany Vs Attrition')

sns.countplot(data['YearsAtCompany'],hue=data['Attrition'],palette='mako')





plt.subplot(212)

plt.title('YearsAtCompany Vs MonthlyIncome')

sns.boxplot(data['YearsAtCompany'],data['MonthlyIncome'])
print('The % of Attrition for employees carrying 0-5 years at company respectively: ')

print(len(data.loc[(data['YearsAtCompany']==0) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['YearsAtCompany']==1) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['YearsAtCompany']==2) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['YearsAtCompany']==3) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['YearsAtCompany']==4) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['YearsAtCompany']==5) & (data['Attrition']=='Yes')])/data.shape[0]*100)
plt.figure(figsize=(20,12))

plt.subplot(211)

plt.title('TotalWorkingYears Vs Attrition')

sns.countplot(data['TotalWorkingYears'],hue=data['Attrition'],palette='prism_r')





plt.subplot(212)

plt.title('TotalWorkingYears Vs MonthlyIncome')

sns.boxplot(data['TotalWorkingYears'],data['MonthlyIncome'])
plt.figure(figsize=(20,11))

plt.subplot(212)

plt.title('NumCompaniesWorked Vs MonthlyIncome')

sns.boxplot(data['NumCompaniesWorked'],data['MonthlyIncome'],hue=data['Attrition'])



#plt.figure(figsize=(15,12))

plt.subplot(211)

plt.title('NumCompaniesWorked Vs Attrition')

sns.countplot(data['NumCompaniesWorked'],hue=data['Attrition'],palette='prism_r')

data.head()
print('The % of Attrition for employees carrying 0-3 years at company respectively: ')

print(len(data.loc[(data['TotalWorkingYears']==0) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['TotalWorkingYears']==1) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['TotalWorkingYears']==2) & (data['Attrition']=='Yes')])/data.shape[0]*100)

print(len(data.loc[(data['TotalWorkingYears']==3) & (data['Attrition']=='Yes')])/data.shape[0]*100)
pd.pivot_table(data=data,index=['TotalWorkingYears'],values=['MonthlyIncome'],aggfunc='mean').sort_values(by='MonthlyIncome').style.set_properties(**{'background-color': 'black',

                                                                                                               'color': 'lawngreen',})  
from scipy.stats import chi2_contingency,chisquare,f_oneway
cat_cols = data.describe(include = "O").columns    # Statistical analysis for categorical data types, Chisquare is performed 
chi_stat=[]

p_value=[]

for i in cat_cols:

    chi_res=chi2_contingency(np.array(pd.crosstab(data[i],data['Attrition'])))

    chi_stat.append(chi_res[0])

    p_value.append(chi_res[1])

chi_square=pd.DataFrame([chi_stat,p_value])

chi_square=chi_square.T

col=['Chi Square Value','P-Value']

chi_square.columns=col

chi_square.index=cat_cols
chi_square
chi_square[chi_square["P-Value"]<0.05]
features_p = list(chi_square[chi_square["P-Value"]<0.05].index)

print("Significant categorical Features:\n\n",features_p)
num_cols = data.describe().columns   # statistical analysis for numerical data dtypes , Therefore performing ANOVA Test
f_stat=[]

p_val=[]

for i in num_cols:

    atr_0=data[data['Attrition']=="No"][i]

    atr_1=data[data['Attrition']=="Yes"][i]

    a=f_oneway(atr_0,atr_1)

    f_stat.append(a[0])

    p_val.append(a[1])

anova=pd.DataFrame([f_stat,p_val])

anova=anova.T

cols=['F-STAT','P-VALUE']

anova.columns=cols

anova.index=num_cols
anova
anova[anova["P-VALUE"]<0.05]
features_p_n = list(anova[anova["P-VALUE"]<0.05].index)

print("Significant numerical Features:\n\n",features_p_n)
data.info()
df=data.copy()
df['Attrition']=df['Attrition'].replace({'Yes':1,'No':0})

df['OverTime']=df['OverTime'].replace({'Yes':1,'No':0})# Repalacing as 0 and 1 for model understanding 
cat_cols=df.select_dtypes('object').columns

cat_cols
for col in cat_cols:

    freqs = df[col].value_counts()

    k = freqs.index[freqs>20][:-1]                 # does the work of One Hot Encoding

    for cat in k:

        name = col+'_'+cat

        df[name] = (df[col] == cat).astype(int)

    del df[col]

    print(col)
df.shape
df.head()
corr=df.corr()

cols=corr.nlargest(15,'Attrition').index

cm = np.corrcoef(df[cols].values.T)

plt.figure(figsize=(20,12))

sns.heatmap(cm,annot=True, yticklabels = cols.values, xticklabels = cols.values, mask = np.triu(cm))
plt.rcParams['figure.figsize'] = (10, 10.0)

df.corr()['Attrition'].sort_values().plot(kind = "barh")
df.head()
df.columns
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

cols=list(df.columns)

cols.remove('Attrition')

for col in cols:

    df[col]=df[col].astype(float)

    df[[col]]=ss.fit_transform(df[[col]])

df['Attrition']=pd.to_numeric(df['Attrition'],downcast='integer')
df.head()
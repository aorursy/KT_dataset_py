# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import *

import squarify 

plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',header=0,skiprows=[1])

text = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv',header=0,skiprows=[1])

question = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv') # This is the collection of questions

schema = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv',index_col=0,header=0,skiprows=[1],nrows=1)
data.Q1.value_counts().nlargest(1)
data.Q1.value_counts().nsmallest(1)
plt.figure(figsize=(15,4))

sns.countplot(data.Q1.sort_values(),edgecolor='black',linewidth=0.5,palette="Blues")

plt.title('Age group of kagglers',fontsize=20)

plt.show()
sns.countplot(data.Q2)

plt.title('Gender')

plt.show()

print(100*data.Q2.value_counts()/len(data.Q2),'%')
plt.figure(figsize=(15,4))

data.Q3[data.Q3.str[:14]=='United Kingdom']='UK'

data.Q3[data.Q3.str[:13]=='United States']='USA'



country_count = data.Q3.value_counts()

country_count.nlargest(10).plot(kind='barh')

plt.title('Country',fontsize=20)

plt.show()



print('Percentage responds from India',round(100*country_count['India']/19717,2),'%')

print('Percentage responds from United States',round(100*country_count['USA']/19717,2),'%')
India = data[data.Q3=='India'][['Q1','Q2','Q3','Q4']]

plt.figure(figsize=(15,7))

plt.subplot('121')

sns.lineplot(India.Q1.value_counts().index,India.Q1.value_counts()/len(India)*100,color='red',marker='o')

sns.lineplot(data.Q1.value_counts().index,data.Q1.value_counts()/len(data)*100,color='blue',marker='o')

plt.ylabel('Percentage respondents')

plt.title('Age group',fontsize =20)

plt.legend(['India','Whole World'])



plt.subplot('122')



labels = data.Q2.value_counts().index

x = np.arange(len(labels))

width = 0.3



plt.bar(x-width/2,data.Q2.value_counts()*100/len(data),width)

plt.bar(x+width/2,India.Q2.value_counts()*100/len(India),width)

plt.legend(['Whole World','India'])

plt.title('Gender',fontsize =20)

plt.ylabel('Percentage respondants')

plt.xticks(x,list(labels))



plt.suptitle('India vs Whole world',fontsize=25)

plt.show()

print(100*India.Q2.value_counts()/len(India.Q2))

plt.show()
vals=pd.pivot_table(India,columns='Q2',index='Q1',values='Q3',aggfunc='count').fillna('0')

vals.drop(['45-49','50-54','55-59','60-69','70+'],inplace=True) #categories with least values are dropped for better visualization

vals.drop(columns=['Prefer not to say','Prefer to self-describe'],inplace=True)

fig,ax=plt.subplots(figsize=(10,10))

size=0.2

ax.pie(vals.sum(axis=1), radius=1,colors=['violet','indigo','b','g','y','orange'],labels=vals.index,

       wedgeprops=dict(width=size, edgecolor='w'))

in_=vals.values.flatten()

ax.pie(in_, radius=1-size,colors=['black','grey'],autopct=lambda x:'{:1.2f}%,({:1.0f})'.format(x,x*sum(in_)/100),

       wedgeprops=dict(width=size, edgecolor='w'))

plt.legend(loc='upper right')

plt.text(-0.18,-0.05,'INDIA',fontsize=36)

plt.text(-1,-1.2,'Note:Male(grey) and Female(black) in the age between 18 to 44 is considered for better visualization')

plt.show()
#countries

print('Number of Countries having respondants more than Indian boys with 18-21 age:', sum(country_count > 1160))

print('Number of Countries having respondants more than Indian girls with 18-21 age:', sum(country_count > 227))
male=data[data.Q2=='Male']

female=data[data.Q2=='Female']

female_male_ratio = female.Q3.value_counts()/male.Q3.value_counts()

fm_perct = round(female_male_ratio*100,2)

plt.figure(figsize=(15,7))



plt.subplot('121')

fm_perct.nlargest(10).plot(kind='bar',color='#12ff56')

plt.title('Large Female - Male Ratio')



plt.subplot('122')

fm_perct.nsmallest(10).plot(kind='bar',color='#ff2460')

plt.title('Lesser Female - Male Ratio')



plt.suptitle('Percentage Male to Female',fontsize=30)

plt.show()
plt.figure(figsize=(15,5))

schema.columns=pd.Series(schema.columns.str.replace('Q','')).apply(int)

schema=schema.T.sort_index()

plt.plot(schema.index,schema)

plt.plot(schema.index,schema,'ro')

plt.xlabel('Question Number')

plt.ylabel('Number of Respondents')

plt.title('Number of Respondents',fontsize=20)

del schema
#Checking Number of respondent

def plot_responds(dat):

    """Check the number of Respondents"""

    plt.pie([sum(dat.notnull()),sum(dat.isnull())],labels=['','No response'],explode=[0,0.03],

            colors=['#12ff56','#ff5470'],shadow=True,autopct='%1.2f%%',radius=0.4)



def plot_single(dat,title,rot=0):

    """Plot the count for single choice question """

    if(len(dat.unique())<4):

        counts=dat.dropna().value_counts()

        fig=plt.pie(counts,autopct='%1.2f%%')

    elif len(dat.unique())<7:

        fig=sns.countplot(dat.sort_values(),palette = "GnBu_d",edgecolor='black',linewidth=0.3)

        fig.set_xticklabels(fig.get_xticklabels(),rotation=rot)

        plt.xlabel(title)

    else:

        fig=sns.countplot(y=dat.sort_values(),palette='GnBu_d',edgecolor='black',linewidth=0.3)

        fig.set_yticklabels(fig.get_yticklabels(),rotation=rot)

        plt.ylabel(title)

    plt.title(title,fontsize=20)

    

    

def plot_multi(QC1,QC2,title,data=data,c='#24bfff'):

    """Plot the count for multi choice question """

    counts=pd.Series()

    for col in data.columns[QC1:QC2+1]:

        key=data[col].dropna().unique()[0]

        counts[key] = sum(data[col].notnull())

        

    

    plt.barh(counts.index,counts,color=c)

    plt.title(title,fontsize=20)

    
question.iloc[0,4]
data.Q4[data.Q4.str[:4]=='Some']='Some College'

data.Q4[data.Q4.str[:9]=='No formal']='High School' #processing for text

plot_single(data.Q4,'Educational Qualification')

plt.title('Education')

plt.show()
Masters = data[data.Q4.str[:6]=="Master"]

Doc = data[data.Q4.str[:8]=='Doctoral']

plt.figure(figsize=(15,10))



plt.subplot('221')

sns.countplot(Masters.Q1.sort_values())

plt.xlabel('Age group')

plt.ylabel('Master degree holders')

plt.title('Master - Age')



plt.subplot('222')

Masters.Q3.value_counts().nlargest(10).plot('barh')

plt.xlabel('Number of Masters')

plt.title('Master -Country')



plt.subplot('223')

sns.countplot(Doc.Q1.sort_values())

plt.xlabel('Age group')

plt.ylabel('Doctoral degree holders')

plt.title('Doctoral - Age')



plt.subplot('224')

Doc.Q3.value_counts().nlargest(10).plot('barh')

plt.xlabel('Number of Doctorals')

plt.title('Doctoral - Country')



plt.suptitle('Educational Qualification',fontsize=25)

plt.show()
plt.figure(figsize=(15,5))

plt.subplot('121')

plot_responds(data.Q5)



plt.subplot('122')

plot_single(data.Q5,'Designation')

plt.suptitle('Designation',fontsize=20)

plt.show()
print('Total Data Scientists:',sum(data.Q5=='Data Scientist'))



print()



print('Total Students:',sum(data.Q5=='Student'))

plt.figure(figsize=(15,5))

plt.subplot('121')

students=data[data.Q5=='Student']

s1=students.Q4.value_counts().sort_index()

s1.index=pd.Series(s1.index).apply(lambda x:str(x).split(' ')[0])

squarify.plot(s1,label=s1.index,alpha=.7,color=['violet','indigo','b','g','y','orange','r'])

plt.title('Students',fontsize=20)



plt.subplot('122')

data_scientist=data[data.Q5=='Data Scientist']

s2=data_scientist.Q4.value_counts().sort_index()

s2.index=pd.Series(s2.index).apply(lambda x:str(x).split(' ')[0])

squarify.plot(s2,label=s2.index,alpha=.7,color=['violet','indigo','b','g','y','orange','r'])

plt.title('Data Scientist',fontsize=20)





plt.show()

del s1,s2
plt.figure(figsize=(15,4))

software=data[data.Q5=='Software Engineer']

soft_perct=software.Q3.value_counts().nlargest(6).div(len(software)).mul(100)

soft_perct.plot(kind='barh',color=['#fe1534','pink','pink','pink','pink','pink'])

plt.title('Software Engineer Location',fontsize=20)

plt.show()
#getting lower limit

def clean_employee(emp):

    """String formating for number of employees"""

    emp=str(emp).replace('>','').replace('employees','')

    emp=emp.split('-')[0].strip()

    return int(''.join(emp.split(',')))





plt.figure(figsize=(15,4))

plt.subplot('121')

plot_responds(data.Q6)

plt.subplot('122')



no_emp = data.Q6.dropna().apply(clean_employee)

plot_single(no_emp,'Employee count')





plt.suptitle('Number of employees')

plt.show()
def plot_pivot(df):

    """visualizing a pivoted table"""

    data=pd.DataFrame()

    for col in df.columns:

        for ind in df.index:

            row=pd.Series([col,ind,df.loc[ind,col]]) #storing x,y,value at x,y

            data=pd.concat([data,row],axis=1)

    data = data.T

    data.columns = [df.columns.name,df.index.name,'count']

    plot=plt.scatter(data.iloc[:,0],data.iloc[:,1],c=data['count'],s=2500,\

                     cmap='rainbow',edgecolor='white',linewidth=1.5)

    #with size=2000 and colored based on value

    plt.colorbar(plot)

    for x,y in data.iterrows():

        plt.annotate(y[2],(y[0],y[1]),color='black')

pv=data.pivot_table(columns='Q6',index='Q5',values='Q1',aggfunc='count')

plt.figure(figsize=(12,12))

plot_pivot(pv) #One can also use heat map at this place

plt.xlabel('Number of Employees')

plt.title('Number of Employees and their designations')

plt.ylabel('')

plt.show()



def clean_data_sci(datasci):

    datasci=str(datasci).split('-')[0].replace('+','')

    return int(datasci)





plt.figure(figsize=(15,5))

plt.subplot('121')

plot_responds(data.Q7)

plt.subplot('122')



no_emp = data.Q7.dropna().apply(clean_data_sci)

sns.countplot(no_emp)

plt.xlabel('Personels employed for Data Science - Lower bound')

plt.title('Counts')



plt.suptitle('Number of Data Science professionals in a company',fontsize=20)

plt.show()
pv=data.pivot_table(columns='Q6',index='Q7',values='Q1',aggfunc='count')

plt.figure(figsize=(14,10))

plot_pivot(pv)

plt.title('Number of Data Science professional vs total employees')

plt.ylabel('')

plt.show()
pv=data.pivot_table(columns='Q7',index='Q5',values='Q1',aggfunc='count')

plt.figure(figsize=(12,12))

plot_pivot(pv)

plt.title('Number of Data Science professional vs respondent\'s designation')

plt.ylabel('')

plt.show()
data.Q8.value_counts().plot(kind='barh')

plt.show()
plot_responds(data.Q8)

plt.title('Response')

plt.show()
pv=data.pivot_table(columns='Q5',index='Q8',values='Q1',aggfunc='count')

plt.figure(figsize=(14,14))

plot_pivot(pv)

plt.title('ML implementation vs respondent\'s designation')

plt.ylabel('')

plt.show()
activities=pd.Series()

activities['Data Analysis'] = sum(data.Q9_Part_1.notnull())

activities['Build Data Infrastructure'] = sum(data.Q9_Part_2.notnull())

activities['Build prototypes'] = sum(data.Q9_Part_3.notnull())

activities['Build ML'] = sum(data.Q9_Part_4.notnull())

activities['Improving ML'] = sum(data.Q9_Part_5.notnull())

activities['Reserach'] = sum(data.Q9_Part_6.notnull())

activities['None'] = sum(data.Q9_Part_7.notnull())

activities['Other'] = sum(data.Q9_Part_8.notnull())


activities_=pd.Series()

activities_['Data Analysis'] = sum(data_scientist.Q9_Part_1.notnull())

activities_['Build Data Infrastructure'] = sum(data_scientist.Q9_Part_2.notnull())

activities_['Build prototypes'] = sum(data_scientist.Q9_Part_3.notnull())

activities_['Build ML'] = sum(data_scientist.Q9_Part_4.notnull())

activities_['Improving ML'] = sum(data_scientist.Q9_Part_5.notnull())

activities_['Reserach'] = sum(data_scientist.Q9_Part_6.notnull())

activities_['None'] = sum(data_scientist.Q9_Part_7.notnull())

activities_['Other'] = sum(data_scientist.Q9_Part_8.notnull())



plt.figure(figsize=(15,7))

plt.subplot('121')

plt.barh(activities.index,activities)

plt.title('General activities',fontsize=20)



plt.subplot('122')

plt.barh(activities_.index,activities_)

plt.title('Data scientist activities',fontsize=20)

plt.ylabel(None)

plt.show()
def clean_salary(sal):

    strTemp = str(sal).split('-')[0]

    if not strTemp[0].isnumeric():

        strTemp =strTemp[1:]

    strTemp=strTemp.replace('$','').replace(' (USD)','')

    cleaned = int(''.join(strTemp.split(',')))

    return cleaned



plt.figure(figsize=(20,10))

plt.subplot('141')

plot_responds(data.Q10)



plt.subplot('142')

salary = data.Q10.dropna().apply(clean_salary)

sns.countplot(y=salary,palette='RdYlGn',linewidth=0.3, edgecolor='black')

plt.title('Overall Salary')



plt.subplot('143')

salary_ds = data_scientist.Q10.dropna().apply(clean_salary)

sns.countplot(y=salary_ds,palette='RdYlGn',linewidth=0.3, edgecolor='black')

plt.title('Data scientist Salary')



print('Total number of students answered:',sum(students.Q10.notnull()))

# also check here 

#print(sum(data[data.Q5=='Student'].Q10.notnull()))



plt.subplot('144')

salary_youth = data[data.Q1=='18-21'].Q10.dropna().apply(clean_salary)

sns.countplot(y=salary_youth,palette='RdYlGn',linewidth=0.3, edgecolor='black')

plt.title('Salary of 18-21 age group')



plt.show()
India=data[data.Q3=='India']

USA=data[data.Q3=='USA']



sal_ind = India.Q10.dropna().apply(clean_salary)

sal_USA = USA.Q10.dropna().apply(clean_salary)



plt.figure(figsize=(15,5))

plt.plot(salary.value_counts().index,salary.value_counts(),'ro')

plt.plot(sal_ind.value_counts().index,sal_ind.value_counts(),'gX')

plt.plot(sal_USA.value_counts().index,sal_USA.value_counts(),'bD')

plt.legend(['General','India','USA'])

plt.title('Salary in India vs US vs all')

plt.xlabel('Salary (in USD)')

plt.show()
# Data scientiat with salary equal to 0-999 are chosen as DS_Sal_low 

DS_Sal_low_index=salary_ds[salary_ds==0].index

DS_Sal_low=data.loc[DS_Sal_low_index]



plt.figure(figsize=(20,12))



plt.subplot('221')

plot_single(DS_Sal_low.Q1,'Age Group')



plt.subplot('222')

(DS_Sal_low.Q3.value_counts()/len(DS_Sal_low)).mul(100).nlargest(10).plot(kind='barh')

plt.title("Countries",fontsize=20)

plt.xlabel('Percentage distribution of countries')



# Indian Data Scientists with low salary

Ind_DS_Sal_low=DS_Sal_low[DS_Sal_low.Q3=='India']



plt.subplot('223')

plot_single(DS_Sal_low.Q6,' No. of employees in Companies with Data Scientist receiving lower salary')



plt.subplot('224')

plot_single(data.Q6,'Number of employees overall')





plt.suptitle("Data Scientists with salary of $ 0-999",fontsize=40)

plt.show()

print('Data Scientist with low salary working in smaller companies(0-49):',sum(DS_Sal_low.Q6=='0-49 employees'))

print('Total Data Scientist with low salary:',len(DS_Sal_low))
plt.figure(figsize=(15,5))

plt.subplot('121')

plot_responds(data.Q10)

plt.subplot('122')



expend_ML_cloud = data.Q11.dropna().apply(clean_salary)

sns.countplot(expend_ML_cloud,palette='Blues',linewidth=0.5,edgecolor='black')

plt.title('Companies count')

plt.suptitle('Expenditure on ML ',fontsize=25)

plt.show()
plt.figure(figsize=(15,5))

plt.subplot('121')

expend_ML_cloud = USA.Q11.dropna().apply(clean_salary)

sns.countplot(expend_ML_cloud,palette='Blues',linewidth=0.5,edgecolor='black')

plt.title('USA Companies count')

plt.xlabel('Expenditure lower bound(in USD)')



plt.subplot('122')

expend_ML_cloud = India.Q11.dropna().apply(clean_salary)

sns.countplot(expend_ML_cloud,palette='Blues',linewidth=0.5,edgecolor='black')

plt.title('India Companies count')

plt.xlabel('Expenditure lower bound(in USD)')



plt.suptitle('Expenditure on ML & CLoud',fontsize=20)

plt.show()
plot_multi(22,33,'Favorite Media')

plt.show()
plt.figure(figsize=(15,8))



plt.subplot('211')

plot_multi(22,33,'Favorite Media of Students', data=students)



plt.subplot('212')

plot_multi(22,33,'Favorite Media of Data Scientist', data=data_scientist,c='r')
plt.figure(figsize=(18,15))

plt.subplot('321')

plot_multi(35,46,'Platforms to begin')

plt.subplot('322')

plot_multi(35,46,'Platforms to begin -Data Scientist',data_scientist,c='r')

plt.subplot('323')

plot_multi(35,46,'Platforms to begin - Data Analyst',data[data.Q5=='Data Analyst'],c='g')

plt.subplot('324')

plot_multi(35,46,'Platforms to begin - Software Engineer',data[data.Q5=='Software Engineer'],c='y')

plt.subplot('325')

plot_multi(35,46,'Platforms to begin - Masters',Masters,c='orange')

plt.subplot('326')

plot_multi(35,46,'Platforms to begin - Doctoral',Doc,c='black')
def check_stat_tool(dat):

    """String Manipulation for basic statistics tool"""

    if 'EXCEL' in str(dat).upper() or 'XL' in str(dat).upper() or 'MICROSOFT' in str(dat).upper() or 'EXEL' in str(dat).upper():

        return 'Excel'

    elif 'GOOGLE' in str(dat).upper() or 'SHEET' in str(dat).upper():

        return 'Google Sheets'

    elif 'PYTHON' in str(dat).upper() or 'PANDA'in str(dat).upper():

        return 'Python'

    elif 'nan' in str(dat):

        return np.NaN

    else:

        return str(dat)

    

basic_stat_tool=text.Q14_Part_1_TEXT.apply(check_stat_tool)

basic_stat_tool=basic_stat_tool.dropna().value_counts().sort_values(ascending=False)[:3]

sns.barplot(basic_stat_tool.index,basic_stat_tool,palette='Blues',linewidth=0.5,edgecolor='black')

plt.title('Basic Statics Tools',fontsize=20)

plt.show()
def check_adv_stat(dat):

    """String Manipulation for Advanced statistics tool"""

    if 'SAS' in str(dat).upper():

        return 'SAS'

    elif 'SPSS' in str(dat).upper() or 'IBM' in str(dat).upper():

        return 'SPSS'

    elif 'PYTHON' in str(dat).upper() or 'PANDA'in str(dat).upper():

        return 'Python'

    elif 'r' in str(dat).upper() or 'R' in str(dat).upper():

        return 'R'

    elif 'MATLAB' in str(dat).upper():

        return 'MATLAB'

    elif 'STATA' in str(dat).upper():

        return 'Stata'

    elif 'NAN' in str(dat).upper():

        return np.NaN

    else:

        return str(dat)

    

Adv_stat_tools = text.Q14_Part_2_TEXT.apply(check_adv_stat).value_counts()[:6]

sns.barplot(Adv_stat_tools.index,Adv_stat_tools)

plt.title('Advanced Statics Tools')

plt.show()
def check_BI(dat):

    """String Manipulation for basic statistics tool"""

    if 'TABL' in str(dat).upper():

        return 'Tableau'

    elif 'POWER' in str(dat).upper():

        return 'PowerBI'

    elif 'SALES' in str(dat).upper():

        return 'Salesforce'

    elif 'QLIK' in str(dat).upper():

        return 'Qlik'

    elif 'SPOT' in str(dat).upper():

        return 'Spotfire'

    elif 'NAN' in str(dat).upper():

        return np.NaN

    else:

        return str(dat)

BI_tools=text.Q14_Part_3_TEXT.apply(check_BI).value_counts()[:5]

sns.barplot(BI_tools.index,BI_tools)

plt.title('Business Intelligence Tools')

plt.show()
def check_IDE(dat):

    if 'JUPYTER' in str(dat).upper() or 'JUPITER' in str(dat).upper():

        return 'Jupyter' # Jupyter Notebook and jupyterlab are grouped together

    elif 'R' in str(dat).upper() :

        return 'RStudio' # Note that all words containing 'R' will be altered

    elif 'PYTHON' in str(dat).upper() or 'PANDA' in str(dat).upper() :

        return 'Python'

    elif 'ANACONDA' in str(dat).upper() :

        return 'Anaconda'

    elif 'MATLAB' in str(dat).upper() :

        return 'MATLAB'

    elif 'COLAB' in str(dat).upper() or 'GOOGLE' in str(dat).upper() :

        return 'Google Colab'

    elif 'NAN' in str(dat).upper():

        return np.NaN

    else:

        return str(dat)

IDE=text.Q14_Part_4_TEXT.apply(check_IDE).value_counts()[:4]

sns.barplot(IDE.index,IDE)

plt.show()
def check_cloud(dat):

    if 'AWS' in str(dat).upper() or 'AMAZON' in str(dat).upper() or 'SAGEMAKER' in str(dat).upper():

        return 'AWS'

    elif 'GCP' in str(dat).upper() or 'GOOGLE' in str(dat).upper()or 'COLAB' in str(dat).upper() or 'BIG' in str(dat).upper() or 'KAGGLE' in str(dat).upper():

        return 'GCP' #kaggle,google colab runs on GCP

    elif 'AZURE' in str(dat).upper() or 'MS' in str(dat).upper() or 'MICROSOFT' in str(dat).upper():

        return 'Azure'

    elif 'DATABRICK' in str(dat).upper():

        return 'DataBricks'

    elif 'WATSON' in str(dat).upper() or 'IBM' in str(dat).upper():

        return 'IBM Cloud'

    elif 'NAN'in str(dat).upper():

        return np.NaN

    else:

        return str(dat)

cloud_tool=text.Q14_Part_5_TEXT.apply(check_cloud).value_counts()[:5]

sns.barplot(cloud_tool.index,cloud_tool)

plt.title('Cloud Services used')

plt.show()
plt.figure(figsize=(17,5))

plt.subplot('121')

plot_responds(data.Q15)

plt.subplot('122')



code_experience = data.Q15.dropna()

sns.countplot(y=code_experience)

plt.title('Companies count')

plt.suptitle('Coding Experience')

plt.show()
plot_multi(56,67,'IDE')
plot_multi(69,80,'Notebook')
plot_multi(82,93,'Programming Language Used')
plot_multi(82,93,'Doctoral',Doc)
plt.figure(figsize=(17,5))

plt.subplot('121')

plot_responds(data.Q19)

plt.subplot('122')



prog_lang_recommendation = data.Q19.dropna()

plot_single(prog_lang_recommendation,'Programming lang')

plt.suptitle('Programming language recommendation')

print(prog_lang_recommendation.value_counts().nlargest(3))

plt.show()
plot_multi(97,108,'Visualization')
plot_multi(110,114,'Hardware Used')
plot_multi(110,114,'Hardware used by companies more than 500,000 USD',data[data.Q10=='> $500,000'],c='r')
def clean_TPU(duration):

    duration=str(duration).replace('>','').replace('times','')

    duration=str(duration).replace('Never','0').replace('Once','1')

    duration=duration.split('-')[0].strip()

    return int(duration)



plt.figure(figsize=(15,5))

plt.subplot('121')

plot_responds(data.Q22)

plt.subplot('122')

TPU = data.Q22.dropna().apply(clean_TPU)

sns.countplot(TPU.sort_index())

plt.xlabel('Number of times used')

plt.title('TPU Usage')



plt.show()
def clean_exp(duration):

    duration=str(duration).replace('+','').replace('years','').replace('< 1','0')

    duration=duration.split('-')[0].strip()

    return int(duration)



plt.figure(figsize=(15,5))

plt.subplot('121')

plot_responds(data.Q23)

plt.subplot('122')



exp=data.Q23.dropna().apply(clean_exp).sort_index()

sns.countplot(exp)

plt.xlabel('Experience (in years)')

plt.suptitle('Experience in ML Methods')

plt.show()
plot_multi(118,129,'ML algorithm regularly used')
plot_multi(118,129,'ML algorithm regularly used by Data Scientist',data_scientist,c='r')
plot_multi(131,138,'Machine Learning Category regularly used')
plot_multi(140,146,'Computer vision category regularly used')
plot_multi(148,153,'NLP Method regularly used')
plot_multi(155,166,'Machine Learning Framework generally used')
plot_multi(168,179,'Cloud computing platform')
plot_multi(181,192,'Cloud Products')
plot_multi(194,205,'Big data')
question.iloc[0,32]
plot_multi(207,218,'Machine Learning products regularly used')
plot_multi(220,231,'Auto ML regularly used')
plot_multi(233,244,'Relational Database regularly used')
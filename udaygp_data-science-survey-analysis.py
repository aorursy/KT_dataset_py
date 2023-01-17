yourFilePath="../input/multipleChoiceResponses.csv"

curencyConverion="../input/conversionRates.csv"

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('seaborn-paper')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import base64

import io

from scipy.misc import imread

import codecs

from IPython.display import HTML

from matplotlib_venn import venn2

from subprocess import check_output



from __future__ import print_function

from ipywidgets import interact, interactive, fixed, interact_manual

from IPython.display import display

import ipywidgets as widgets

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score , accuracy_score, roc_curve, auc

from sklearn import tree

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.neighbors import KNeighborsClassifier 



import operator

import os,sys

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

from pandas.api.types import CategoricalDtype

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder, LabelBinarizer



from sklearn_pandas import DataFrameMapper



def f(x):

    return x



def loadData(yourFilePath):

    df=pd.read_csv(yourFilePath, low_memory=False,encoding='ISO-8859-1')

    return df



def summaryColumn(columnName):

    column_summary=df_raw[[columnName]].groupby([columnName]).size().reset_index(name='counts')

    column_summary=column_summary.sort_values(['counts'],ascending=[0])

    column_category=list(column_summary[columnName])

    category_counts=list(column_summary['counts'])

#     plt.rcdefaults()

    fig, ax = plt.subplots()

    y_pos=np.arange(len(column_category))

    ax.barh(y_pos, category_counts, align='center',

        color='green')

    ax.set_yticks(y_pos)

    ax.set_yticklabels(column_category)

    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_xlabel('Count')

    ax.set_title(columnName)

    plt.show()

    return column_summary
data = loadData(yourFilePath)

rates = loadData(curencyConverion)

df_raw=loadData(yourFilePath)

columnList=list(df_raw.columns)
result=interactive(summaryColumn,columnName=columnList);

display(result)
# Major and CurrentJobTitle Relation   

f,ax=plt.subplots(1,2,figsize=(20,10))                                                                                  

sns.countplot(y = data['MajorSelect'],ax=ax[0],order= data['MajorSelect'].value_counts().index)                         

ax[0].set_title('Major')                                                                                                

ax[0].set_ylabel('')                                                                                                    

sns.countplot(y= data['CurrentJobTitleSelect'],ax=ax[1],order=data['CurrentJobTitleSelect'].value_counts().index)       

ax[1].set_title('Current Job')                                                                                          

ax[1].set_ylabel('')                                                                                                    

plt.subplots_adjust(wspace=0.8)                                                                                         

plt.show()   
  

work_tools = data['WorkToolsSelect'].dropna().str.split(',')              

tools = []                                                                

for wktools in work_tools:                                                

    for tool in wktools:                                                  

        tools.append(tool)                                                

result = pd.Series(tools).value_counts()[:10]                             

plt.subplots(figsize=(10,10))                                             

sns.barplot(result.values,result.index)                                   

plt.title('Work Tools')                                                   

plt.show()              
resp = data.dropna(subset=['WorkToolsSelect'])                                                                

resp = pd.merge(resp,rates,left_on='CompensationCurrency',right_on='originCountry',how='left')                

python = resp[(resp['WorkToolsSelect'].str.contains('Python'))&(~resp['WorkToolsSelect'].str.contains('R'))]  

R = resp[(~resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]       

both = resp[(resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))] 

# python and R users recommendations:                                                

p_reconmd = python['LanguageRecommendationSelect'].value_counts()[:2]                

r_reconmd = R['LanguageRecommendationSelect'].value_counts()[:2]                     

labels1 = p_reconmd.index                                                            

values1 = p_reconmd.values                                                           

labels2 = r_reconmd.index                                                            

values2 = r_reconmd.values                                                           

f,ax = plt.subplots(1,2,figsize=(10,10))                                             

ax[0].pie(values1, labels = labels1,autopct='%1.1f%%', shadow=False, startangle=90)  

ax[0].axis('equal')                                                                  

ax[0].set_title('Python Users Recommendation')                                       

ax[1].pie(values2, labels = labels2,autopct='%1.1f%%', shadow=False, startangle=90)  

ax[1].axis('equal')                                                                  

ax[1].set_title('R Users Recommendation')                                            

plt.show()      
#  python and R salary compare:  

py_sal=(pd.to_numeric(python['CompensationAmount'].dropna(),errors='coerce')*python['exchangeRate']).dropna()

py_avr_sal = pd.Series(py_sal).median()



R_sal=(pd.to_numeric(R['CompensationAmount'].dropna(),errors='coerce')*R['exchangeRate']).dropna()

R_avr_sal = pd.Series(R_sal).median()



both_sal=(pd.to_numeric(both['CompensationAmount'].dropna(),errors='coerce')*both['exchangeRate']).dropna()

both_avr_sal = pd.Series(both_sal).median()

print ('Median Salary For Individual using Python:',py_avr_sal)

print ('Median Salary For Individual using R:',R_avr_sal)

print ('Median Salary For Individual knowing both languages:',both_avr_sal)
                                                                      

salary=data[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()   

salary=pd.merge(salary,rates,left_on='CompensationCurrency',right_on='originCountry',how='left')                       

                                                                                                                       

salary['Salary'] = pd.to_numeric(salary['CompensationAmount'],errors='coerce') * salary['exchangeRate'].dropna()       

salary_null = pd.isnull(salary['Salary'])                                                                              

                                                                                                                       

salary_null_false= salary['Salary'][salary_null == False][salary['Salary'] >= 0]                                       



#Compensation By Job Title   

sal_job = salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame().sort_values(by='Salary',ascending=False)

plt.subplots(figsize=(10,10))                                                                                           

sns.barplot(sal_job.Salary,sal_job.index)                                                                               

plt.title('Compensation By Job Title')                                                                                  

plt.show()                   


employed=['Employed full-time','Independent contractor, freelancer, or self-employed','Employed part-time']

df_employed=df_raw[df_raw['EmploymentStatus'].isin(employed)]

df_employed[['CodeWriter']].groupby(['CodeWriter']).size().reset_index(name='counts')
switcher=df_raw[df_raw['CareerSwitcher']=='Yes']

switcher[['CurrentJobTitleSelect']].groupby(['CurrentJobTitleSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])
df_raw['TitleFit_Score']=df_raw['TitleFit'].apply(lambda x: 5 if x=='Perfectly' else (3 if x=='Fine' else 1))

titleFit_idx=df_raw['TitleFit'].isnull()

df_withTitleFit=df_raw[~titleFit_idx]

print(df_withTitleFit.groupby(['CurrentJobTitleSelect'])['TitleFit_Score'].mean())
df_raw_pre=loadData(yourFilePath)

obj_df1 = df_raw_pre.select_dtypes(include=['object']).copy()

obj_df1[obj_df1.isnull().any(axis=1)]



nonCatlistColumns = ['Age','LearningCategorySelftTaught','LearningCategoryOnlineCourses','LearningCategoryWork','LearningCategoryUniversity',

     'LearningCategoryKaggle','LearningCategoryOther','TimeGatheringData',

     'TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights','TimeOtherSelect']



def convertToInt(amt):

    try:

        if ',' in amt:

            return float(amt.replace(',',''))

        elif amt != '-':

            return float(amt)

    except:

        return float(amt)



        

def convertToString(str):

    if '' == str:

        return ''

    else:

        return str

    

def groupSatifaction(val):

    if 0 < val and val < 7:

        return 1

    #elif 5<= val and 7 > val:

     #   return 2

    elif 7<= val:

        return 2

    else:

        return 0

    

compensation = ['CompensationAmount','CompensationCurrency']

comp = obj_df1['CompensationAmount']

compCur = obj_df1['CompensationCurrency']



comp = comp.apply(convertToInt )

compCur = compCur.apply(convertToString)

obj_df1['CompensationAmount'] = comp

    

columns = list(df_raw_pre.columns)

for column in columns:

    if column in nonCatlistColumns:

        continue

    obj_df1[column] = obj_df1[column].astype('category').cat.codes
def featurize():

    return DataFrameMapper(obj_df1.columns)





pipeline = Pipeline([('featurize', featurize()), ('forest', RandomForestClassifier())])



y = obj_df1['JobSatisfaction'].apply(groupSatifaction)

X = obj_df1[obj_df1.columns.drop('JobSatisfaction')]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestClassifier

clf_rfc = RandomForestClassifier(max_depth=2, random_state=0)

clf_rfc.fit(X_train, y_train)



i=0

columns = list(X.columns)

for val in np.nditer(clf_rfc.feature_importances_):

    if val > 0.01 :

        print (val, columns[i])

    i = i+1    



print(clf_rfc.score(X_test, y_test))
df_raw[['MLToolNextYearSelect']].groupby(['MLToolNextYearSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])
df_raw[['MLMethodNextYearSelect']].groupby(['MLMethodNextYearSelect']).size().reset_index(name='counts').sort_values(['counts'],ascending=[0])
learningPlatformSurvey=['LearningPlatformUsefulnessArxiv','LearningPlatformUsefulnessBlogs'

                        ,'LearningPlatformUsefulnessCollege','LearningPlatformUsefulnessCompany'

                        ,'LearningPlatformUsefulnessConferences','LearningPlatformUsefulnessFriends'

                        ,'LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessNewsletters'

                        ,'LearningPlatformUsefulnessCommunities'

                        ,'LearningPlatformUsefulnessDocumentation','LearningPlatformUsefulnessCourses'

                        ,'LearningPlatformUsefulnessProjects','LearningPlatformUsefulnessPodcasts'

                        ,'LearningPlatformUsefulnessSO','LearningPlatformUsefulnessTextbook'

                        ,'LearningPlatformUsefulnessTradeBook','LearningPlatformUsefulnessTutoring'

                        ,'LearningPlatformUsefulnessYouTube']

for surveyTarget in learningPlatformSurvey:

    df_raw['{}_Score'.format(surveyTarget)]=df_raw[surveyTarget].apply(lambda x: 5 if x=='Very useful' else (3 if x=='Somewhat useful' else 1))



    

# for DataScientist

df_raw_DataScientist=df_raw[df_raw['CurrentJobTitleSelect']=='Data Scientist']

for surveyTarget in learningPlatformSurvey:

    print ('{} average rate is {}'.format(surveyTarget,df_raw_DataScientist['{}_Score'.format(surveyTarget)].mean()))
# for Software engineer

df_raw_SoftwareEngineer=df_raw[df_raw['CurrentJobTitleSelect']=='Software Developer/Software Engineer']

for surveyTarget in learningPlatformSurvey:

    print ('{} average rate is {}'.format(surveyTarget,df_raw_SoftwareEngineer['{}_Score'.format(surveyTarget)].mean()))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from wordcloud import WordCloud, STOPWORDS

import collections as cl

from datetime import datetime
data = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')
data.head()
data.columns
desc = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')

desc.values
data.drop(columns={'Respondent','Hobbyist','Employment','Student','EdLevel','UndergradMajor','EduOther','OrgSize','DevType','YearsCode','Age1stCode','YearsCodePro','JobSat','MgrIdiot','MgrMoney','MgrWant','JobSeek','LastHireDate','LastInt','FizzBuzz','JobFactors','CurrencySymbol','CurrencyDesc','CompTotal','CompFreq','WorkPlan','WorkChallenge','WorkRemote','ImpSyn','CodeRev','CodeRevHrs','UnitTests','PurchaseHow','PurchaseWhat','MiscTechWorkedWith','MiscTechDesireNextYear','DevEnviron','Containers','BlockchainOrg','BlockchainIs','BetterLife','ITperson','OffOn','Extraversion','ScreenName','SOVisit1st','SOVisitFreq','SOVisitTo','SOFindAnswer','SOTimeSaved','SOHowMuchTime','SOAccount','SOPartFreq','SOJobs','EntTeams','SOComm','WelcomeChange','OpenSource','SONewContent','Trans','Sexuality','Ethnicity','Dependents','SurveyLength','SurveyEase'},inplace = True)
data.head()
data.shape
data.info()
data.describe()
data['MainBranch'].value_counts()
MainBranch = []

for i in data['MainBranch']:

    if i == 'I am a developer by profession':

        MainBranch.append('Developer')

    elif i == 'I am a student who is learning to code':

        MainBranch.append('Student')

    elif i == 'I am not primarily a developer, but I write code sometimes as part of my work':

        MainBranch.append('Semi_Developer')

    elif i == 'I code primarily as a hobby':

        MainBranch.append('Hobby') 

    elif i == 'I used to be a developer by profession, but no longer am':

        MainBranch.append('Ex_Developer')

    else:

        MainBranch.append('Student')

data['MainBranch'] = MainBranch

data['MainBranch'] = data['MainBranch'].astype('category',inplace=True)
data['ResumeUpdate'].value_counts()
data['ResumeUpdate'] = data['ResumeUpdate'].astype('category')
data['OpenSourcer'].value_counts()
OpenSourcer = []

for i in data['OpenSourcer']:

    if (i == 'Never' ) or (i=='Less than once per year') or (i==''):

        OpenSourcer.append('No')

    else:

        OpenSourcer.append('Yes')

data['OpenSourcer'] = OpenSourcer

data['OpenSourcer'] = data['OpenSourcer'].astype('category')
data['CareerSat'].value_counts()
data['CareerSat'] = data['CareerSat'].astype('category')
data['WorkLoc'].value_counts()
WorkLoc = []

for i in data['WorkLoc']:

    if i == 'Office':

        WorkLoc.append(i)

    elif i == 'Home':

        WorkLoc.append('Home')

    else:

        WorkLoc.append('Other')

data['WorkLoc'] = WorkLoc

data['WorkLoc'] = data['WorkLoc'].astype('category')
data['OpSys'].value_counts()
data['OpSys'] = data['OpSys'].astype('category')
Age = []

count = 0

for i in data['Age']:

    try:

        Age.append(int(i))

    except:

        Age.append(i)

        count += 1

data['Age'] = Age
data['Gender'].value_counts()
Gender = []

for i in data['Gender']:

    if (i=='Man') or (i== 'Man;Non-binary, genderqueer, or gender non-conforming'):

        Gender.append('Male')

    elif (i=='Woman') or (i=='Woman;Non-binary, genderqueer, or gender non-conforming') or (i=='Woman;Man;Non-binary, genderqueer, or gender non-conforming'):

        Gender.append('Female')

    else:

        Gender.append('Transgender')

data['Gender'] = Gender

data['Gender'] = data['Gender'].astype('category')
Country = data['Country'].value_counts().index

filters = []

for i in data['Country']:

    filters.append(i in Country)

    

data = data[filters]
data.info()
data.shape
dev = data.groupby('MainBranch')['OpenSourcer'].value_counts()

dev = dev.to_frame('Number_of_Developers')

dev = dev.reset_index()

sns.barplot(x='MainBranch',y='Number_of_Developers',hue='OpenSourcer',data=dev)
num = data['Country'].value_counts()[:30]

total = data.groupby('Country')['ConvertedComp'].sum()

data_plot1 = (total/num).sort_values(ascending=False)[:30]

data_plot = data_plot1.reset_index()

data_plot.rename(columns={0:'Average income in USD','index':'Country Name'},inplace=True)

sns.barplot(y='Country Name',x='Average income in USD',data = data_plot)
data_dev = data[data['MainBranch']=='Developer']

plot_data=data_dev['CareerSat'].value_counts().reset_index()

plot_data.rename(columns={'index':'Amount of Satisfaction','CareerSat':'Number of Developers'},inplace=True)

sns.barplot(y='Amount of Satisfaction',x='Number of Developers',data=plot_data)
plot_data = data_dev['Age'].value_counts().reset_index()

plot_data.rename(columns={'index':'Age','Age':'Number of Developers'},inplace=True)

sns.lineplot(x='Age', y='Number of Developers', data=plot_data)
data_plot = data_dev['SocialMedia'].value_counts().reset_index()

data_plot.rename(columns={'index':'Name of SocialMedia','SocialMedia':'Number of Users'},inplace=True)

sns.barplot(x='Number of Users',y='Name of SocialMedia',data=data_plot)
data_plot = data_dev['OpSys'].value_counts().reset_index()

data_plot.rename(columns={'index':'Name of Operating System','OpSys':'Number of Users'},inplace=True)

sns.barplot(x='Number of Users',y='Name of Operating System',data=data_plot)
data_plot = data_dev['ResumeUpdate'].value_counts().reset_index()

data_plot.rename(columns={'index':'Reason for updating RESUME','ResumeUpdate':'Number of Developers'},inplace=True)

sns.barplot(x='Number of Developers',y='Reason for updating RESUME',data=data_plot)
def generate_word_column_for_the_column_of(column):

    column_name = column

    os_now_all_word = ''

    for i in data_dev[column_name]:

        try:

            a=i.split(';')

            for j in a:

                os_now_all_word+=' '+ j

        except:

            a=5



    cloud = WordCloud(background_color="white",max_font_size=250,width=960, height=1080).generate(os_now_all_word)



    cloud.to_file(column_name + '.png')

    plt.imshow(cloud)
generate_word_column_for_the_column_of('PlatformWorkedWith')
generate_word_column_for_the_column_of('PlatformDesireNextYear')
generate_word_column_for_the_column_of('LanguageWorkedWith')
generate_word_column_for_the_column_of('LanguageDesireNextYear')
generate_word_column_for_the_column_of('DatabaseWorkedWith')
generate_word_column_for_the_column_of('DatabaseDesireNextYear')
generate_word_column_for_the_column_of('WebFrameWorkedWith')
generate_word_column_for_the_column_of('WebFrameDesireNextYear')
data_plot_number = data_dev['WorkLoc'].value_counts().values

data_plot_name = data_dev['WorkLoc'].value_counts().index 

plt.pie(data_plot_number,labels=data_plot_name,autopct='%1.1f%%',)
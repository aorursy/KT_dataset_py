#Get necessory libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/survey_results_public.csv')
data.head()
features_all = data.columns.tolist()
features_all
data.drop(columns=[
# 'Respondent',
#  'Hobby',
#  'OpenSource',
#  'Country',
#  'Student',
#  'Employment',
#  'FormalEducation',
#  'UndergradMajor',
#  'CompanySize',
#  'DevType',
#  'YearsCoding',
#  'YearsCodingProf',
#  'JobSatisfaction',
#  'CareerSatisfaction',
#  'HopeFiveYears',
#  'JobSearchStatus',
#  'LastNewJob',
 'AssessJob1',
 'AssessJob2',
 'AssessJob3',
 'AssessJob4',
 'AssessJob5',
 'AssessJob6',
 'AssessJob7',
 'AssessJob8',
 'AssessJob9',
 'AssessJob10',
 'AssessBenefits1',
 'AssessBenefits2',
 'AssessBenefits3',
 'AssessBenefits4',
 'AssessBenefits5',
 'AssessBenefits6',
 'AssessBenefits7',
 'AssessBenefits8',
 'AssessBenefits9',
 'AssessBenefits10',
 'AssessBenefits11',
 'JobContactPriorities1',
 'JobContactPriorities2',
 'JobContactPriorities3',
 'JobContactPriorities4',
 'JobContactPriorities5',
 'JobEmailPriorities1',
 'JobEmailPriorities2',
 'JobEmailPriorities3',
 'JobEmailPriorities4',
 'JobEmailPriorities5',
 'JobEmailPriorities6',
 'JobEmailPriorities7',
 'UpdateCV',
 'Currency',
#  'Salary',
#  'SalaryType',
#  'ConvertedSalary',
 'CurrencySymbol',
 'CommunicationTools',
 'TimeFullyProductive',
 'EducationTypes',
 'SelfTaughtTypes',
 'TimeAfterBootcamp',
 'HackathonReasons',
 'AgreeDisagree1',
 'AgreeDisagree2',
 'AgreeDisagree3',
 'LanguageWorkedWith',
 'LanguageDesireNextYear',
 'DatabaseWorkedWith',
 'DatabaseDesireNextYear',
 'PlatformWorkedWith',
 'PlatformDesireNextYear',
 'FrameworkWorkedWith',
 'FrameworkDesireNextYear',
#  'IDE',
#  'OperatingSystem',
 'NumberMonitors',
 'Methodology',
 'VersionControl',
 'CheckInCode',
 'AdBlocker',
 'AdBlockerDisable',
 'AdBlockerReasons',
 'AdsAgreeDisagree1',
 'AdsAgreeDisagree2',
 'AdsAgreeDisagree3',
 'AdsActions',
 'AdsPriorities1',
 'AdsPriorities2',
 'AdsPriorities3',
 'AdsPriorities4',
 'AdsPriorities5',
 'AdsPriorities6',
 'AdsPriorities7',
 'AIDangerous',
 'AIInteresting',
 'AIResponsible',
 'AIFuture',
 'EthicsChoice',
 'EthicsReport',
 'EthicsResponsible',
 'EthicalImplications',
 'StackOverflowRecommend',
 'StackOverflowVisit',
 'StackOverflowHasAccount',
 'StackOverflowParticipate',
 'StackOverflowJobs',
 'StackOverflowDevStory',
 'StackOverflowJobsRecommend',
 'StackOverflowConsiderMember',
 'HypotheticalTools1',
 'HypotheticalTools2',
 'HypotheticalTools3',
 'HypotheticalTools4',
 'HypotheticalTools5',
 'WakeTime',
 'HoursComputer',
 'HoursOutside',
 'SkipMeals',
 'ErgonomicDevices',
 'Exercise',
#  'Gender',
#  'SexualOrientation',
#  'EducationParents',
#  'RaceEthnicity',
#  'Age',
#  'Dependents',
#  'MilitaryUS',
#  'SurveyTooLong',
#  'SurveyEasy'
         ],inplace=True)
data.shape
data.columns
data.dtypes
data.describe(include='all')
data.groupby("OperatingSystem").size()
#Percentage of operating system used
percdata = data[data['OperatingSystem'].isin(['Windows','MacOS','Linux-based'])]['OperatingSystem'].value_counts().transform(lambda x: (x/sum(x))*100)
percdata.plot(kind='bar',title='OS Used')
data['Gender'].value_counts()
filterd_gen = data[data['Gender'].isin(["Male", "Female","Transgender"])]
filterd_gen['Gender'].value_counts()
fig, ax = plt.subplots()
sns.countplot(y='Gender',hue='OperatingSystem',data=filterd_gen,ax=ax)
fig.set_size_inches(13,5)
ax.set_title('OS used')
top_ides = data.IDE.value_counts().index[0:21]
top_ides
sns.countplot(y='Age',data=data)
data_IDE = data[data['IDE'].isin(top_ides)]
data[data['IDE'].isin(top_ides)]['Age'].value_counts()
age_IDE = data_IDE.groupby(['Age','IDE']).size()
age_IDE.to_frame()
# age_IDE.to_csv()
# there should be 140 rows coz fro eace age caegory we have 20 more sub categores so 20*7=140
fig1, ax1 = plt.subplots()
sns.countplot(y='IDE',hue='Age',data=data_IDE,ax=ax1)
fig1.set_size_inches(20,18)
ax1.legend(loc='lower right')
ax1.set_title('IDE used')
top_ides5 = data.IDE.value_counts().index[0:5]
data_DE5 = data[data['IDE'].isin(top_ides5)]
age_IDE5 =data_DE5.groupby(['Age','IDE']).size()
fig1, ax1 = plt.subplots()
sns.countplot(x='IDE',hue='Age',data=data_DE5,ax=ax1)
fig1.set_size_inches(19,5)
ax1.legend(loc='best')
ax1.set_title('IDE and Age Analysis')
fig1.savefig('IDE and Age Analysis')
age_IDE.to_frame()
# age_IDE.to_csv()

data.shape
data_IDE = data[['IDE','Age']]
data_IDE.shape
data_IDE.dropna(inplace=True)
data_IDE.shape
for i in data_IDE.IDE.values:
    if ';' in i:
        data_IDE=data_IDE[data_IDE.IDE != i]
data_IDE.head()
data_IDE.IDE.value_counts()
top_ide_fil5 = data_IDE.IDE.value_counts().index[0:10]
data_IDE_fil5 = data_IDE[data_IDE['IDE'].isin(top_ide_fil5)]
age_IDE_fil5 =data_IDE_fil5.groupby(['Age','IDE']).size()
fig2, ax2 = plt.subplots()
sns.countplot(x='IDE',hue='Age',data=data_IDE_fil5,ax=ax2)
fig2.set_size_inches(18,8)
ax1.legend(loc='best')
ax2.set_title('IDE and Age Analysis')
fig2.savefig('IDE and Age Analysis')
sns.set(rc={'figure.figsize':(16,8)})

sns.countplot(x="Age", hue = "JobSatisfaction", data=data).set_title("Age Count of StackOverflow Survey Data")
filtered = data[data["Gender"].isin(["Male","Female"])].dropna(subset = ['Age'])
sns.set(rc={'figure.figsize':(12,8)})
sns.catplot(y="Age", hue = "JobSatisfaction", col = "Gender", data= filtered, kind="count", height=10, aspect = 0.9)
sns.set(style="whitegrid")
sns.violinplot(x="Age", y="ConvertedSalary", data=filtered);

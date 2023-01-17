# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.offline as py
import plotly.express as px

import seaborn as sns
from matplotlib import rcParams
import plotly.graph_objects as go

import matplotlib.pyplot as plt
df = pd.read_csv("../input/covid19-patient-precondition-dataset/covid.csv", parse_dates=['entry_date','date_symptoms'], dayfirst=True)
df
no_icu_data_bool = df['icu'].isin([97, 98, 99])
no_icu_data_bool

icu_data = df[~ no_icu_data_bool]
no_icu_data = df[no_icu_data_bool]
print("{} rows have ICU details ".format(icu_data.shape[0]))
print("Only {}% of given data has ICU details ".format(round((icu_data.shape[0]/ no_icu_data.shape[0])*100)))
icu_data.sex.replace({1: 'Female', 2: 'Male'}, inplace=True)
icu_data.patient_type.replace({1: 'Outpatient', 2: 'Inpatient'}, inplace=True)
icu_data.intubed.replace({1: 'Yes', 2: 'No',97:'Not Specified', 98:'Not Specified',99:'Not Specified'}, inplace=True)
icu_data.pneumonia.replace({1: 'Yes', 2: 'No', 98:'Not Specified',99:'Not Specified', 97:'Not Specified'}, inplace=True)
icu_data.pregnancy.replace({1: 'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
icu_data.diabetes.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified',99:'Not Specified'}, inplace=True)
icu_data.copd.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified', 99:'Not Specified'}, inplace=True)
icu_data.asthma.replace({1: 'Yes', 2: 'No', 99:'Not Specified',97:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.inmsupr.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified',99:'Not Specified'}, inplace=True)
icu_data.hypertension.replace({1: 'Yes', 2: 'No', 99:'Not Specified',97:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.other_disease.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.cardiovascular.replace({1: 'Yes', 2: 'No', 99:'Not Specified',97:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.obesity.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified',99:'Not Specified'}, inplace=True)
icu_data.renal_chronic.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.tobacco.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.contact_other_covid.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)
icu_data.covid_res.replace({1: 'Positive', 2: 'Negative', 3:'Awaiting Results'}, inplace=True)
icu_data.icu.replace({1: 'Yes', 2: 'No', 97:'Not Specified',98:'Not Specified', 99:'Not Specified'}, inplace=True)

icu_yes = icu_data[icu_data['icu'] == "Yes"]
icu_no = icu_data[icu_data['icu'] != "Yes"]
print("In Patients requiring ICU = {} and not requiring ICU = {} ".format(icu_yes.shape[0],icu_no.shape[0]))
icu_yes.date_died = icu_yes.date_died.replace("9999-99-99",datetime.datetime(1900,1,1))
icu_yes.date_died = pd.to_datetime(icu_yes.date_died, dayfirst=True, errors='coerce')
icu_yes['died'] = np.where(icu_yes['date_died'] == datetime.datetime(1900,1,1),False,True)
icu_yes['entry_symptoms_dates'] = ((icu_yes['entry_date'] - icu_yes['date_symptoms'])/ np.timedelta64(1, 'D')).astype(int)
icu_yes['died_entry_dates'] = ((icu_yes['date_died'] - icu_yes['entry_date'])/ np.timedelta64(1, 'D')).astype(int)
icu_yes['died_symptoms_dates'] = ((icu_yes['date_died'] - icu_yes['date_symptoms'])/ np.timedelta64(1, 'D')).astype(int)
icu_yes['age_freq']=np.where((icu_yes['age'] < 2),'<2',np.where((icu_yes['age'] >= 2) & (icu_yes['age'] <= 12) ,'2-12',np.where((icu_yes['age'] >= 13) & (icu_yes['age'] < 18) ,'13-17',np.where(icu_yes['age']<18,'<18',np.where((icu_yes['age']>17)&(icu_yes['age']<=30),'18-30',
np.where((icu_yes['age']>30)&(icu_yes['age']<=50),'31-50',np.where(icu_yes['age']>70,'70+',
np.where((icu_yes['age']>50)&(icu_yes['age']<=70),'51-70',"Not Specified"))))))))

icu_yes.head(4)
def plot_dates(column_name,title):             
    monthly_df = icu_yes.groupby([icu_yes[column_name].dt.to_period("M").astype(str),'sex'])['id'].agg('count').to_frame(name='count').reset_index()   
    if column_name == 'date_died': # removing the first 2 rows because of some faulty data
        monthly_df = monthly_df[2:]        
   
    fig = px.bar(monthly_df, x=column_name, y="count",color="sex",title="Patients realized symptoms")        
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(title_text='{} (Jan-Jun 2020)'.format(title), title_x=0.5,showlegend=True)
    fig.show()
plot_dates('date_symptoms','Patient symptoms arisen')
plot_dates('entry_date','Patient Admissions')
plot_dates('date_died','Patient deaths')
symptoms_by_date = icu_yes.groupby('date_symptoms')['id'].agg('count').to_frame(name='count')
deaths_by_date = icu_yes.groupby('date_died')['id'].agg('count').to_frame(name='count')
entry_by_date = icu_yes.groupby('entry_date')['id'].agg('count').to_frame(name='count')

deaths_by_date = deaths_by_date[1:]
colort = '#456213'
color1 = '#9467bd'
color2 = '#2367ff'
color3 = '#15dd88'

trace1 = go.Scatter(x = deaths_by_date.index, y = deaths_by_date['count'], name='Deaths by Date', line = dict( color = color1))
trace2 = go.Scatter(x = symptoms_by_date.index, y = symptoms_by_date['count'], name='Symptoms by Date', line = dict(  color = color2 ) )
trace3 = go.Scatter(x = entry_by_date.index, y = entry_by_date['count'], name='Entry by Date', line = dict(color = color3 ) )

data = [trace1,trace2,trace3]
layout = go.Layout(title= "Cases by day (Jan- Jun 2020)", yaxis=dict(title='Number of deaths', titlefont=dict(color=colort), tickfont=dict(color=colort)))
fig = go.Figure(data=data, layout=layout)
plot_url = py.iplot(fig)
sdates_df = icu_yes.groupby(['died_symptoms_dates']).agg('count')['id'].to_frame(name='count').reset_index()
entry_bool = sdates_df['died_symptoms_dates'] >= 0
sdates_df = sdates_df[entry_bool]
fig = px.scatter(sdates_df, x="died_symptoms_dates", y="count",
                 color="count", color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title="Number of days patients lived from Symptoms to Death ",title_x=0.5,xaxis=dict(title="Days"),yaxis=dict(title="Number of Patients"))
fig.show()
ddates_df = icu_yes.groupby(['died_entry_dates']).agg('count')['id'].to_frame(name='count').reset_index()
entry_bool = ddates_df['died_entry_dates'] >= 0
ddates_df = ddates_df[entry_bool]
fig = px.scatter(ddates_df, x="died_entry_dates", y="count",
                 color="count", color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title="Number of days patients lived between Admission to Death ",title_x=0.5,xaxis=dict(title="Days"),yaxis=dict(title="Number of Patients"))
fig.show()
dates_df = icu_yes.groupby(['entry_symptoms_dates']).agg('count')['id'].to_frame(name='count').reset_index()
fig = px.scatter(dates_df, x="entry_symptoms_dates", y="count",
                 color="count", color_continuous_scale=px.colors.sequential.Viridis)
fig.update_layout(title="Number of days between - When Symptoms appeared to Hospital Admissions",title_x=0.5,xaxis=dict(title="Days"),yaxis=dict(title="Number of Patients"))
fig.show()
negative_result_bool = (icu_yes['covid_res'] == 'Negative') 
positive_result_bool = (icu_yes['covid_res'] == 'Positive')
awaiting_result_bool = (icu_yes['covid_res'] == 'Awaiting Results') 
pne_bool = (icu_yes['pneumonia'] == 'Yes')
intubed_bool = (icu_yes['intubed'] == 'Yes')
copd_bool = (icu_yes['copd'] == 'Yes') 
tobacco_bool = (icu_yes['tobacco'] == 'Yes')
asthma_bool = (icu_yes['asthma'] == 'Yes')
sex_bool = (icu_yes['sex'] == 'Female') 
preg_bool = (icu_yes['pregnancy'] == 'Yes') 
awaiting_result_bool = (icu_yes['covid_res'] == 'Awaiting Results')
inmsupr_bool = (icu_yes['inmsupr'] == 'Yes')
obesity_bool = (icu_yes['obesity'] == 'Yes')
hypertension_bool = (icu_yes['hypertension'] == 'Yes')
cardiovascular_bool = (icu_yes['cardiovascular'] == 'Yes')
renal_chronic_bool = (icu_yes['renal_chronic'] == 'Yes')
other_disease_bool = (icu_yes['other_disease'] == 'Yes')
diabetes_bool = (icu_yes['diabetes'] == 'Yes')
non_diabetes_bool = (icu_yes['diabetes'] == 'Yes')
died_bool = icu_yes['died'] == True
not_died_bool = icu_yes['died'] == False
fig = go.Figure()
x11 = icu_yes[positive_result_bool].shape[0]
x12 = icu_yes[positive_result_bool & pne_bool].shape[0]
x13 = icu_yes[positive_result_bool & pne_bool & intubed_bool].shape[0]
x14 = icu_yes[positive_result_bool & pne_bool & intubed_bool & tobacco_bool].shape[0]
x15 = icu_yes[positive_result_bool & pne_bool & intubed_bool & copd_bool & tobacco_bool].shape[0]
x16 = icu_yes[positive_result_bool & pne_bool & intubed_bool & asthma_bool & tobacco_bool & copd_bool].shape[0]

x21 = icu_yes[negative_result_bool].shape[0]
x22 = icu_yes[negative_result_bool & pne_bool].shape[0]
x23 = icu_yes[negative_result_bool & pne_bool & intubed_bool].shape[0]
x24 = icu_yes[negative_result_bool & pne_bool & intubed_bool & tobacco_bool].shape[0]
x25 = icu_yes[negative_result_bool & pne_bool & intubed_bool & copd_bool & tobacco_bool].shape[0]
x26 = icu_yes[negative_result_bool & pne_bool & intubed_bool & asthma_bool & tobacco_bool & copd_bool].shape[0]

x31 = icu_yes[awaiting_result_bool].shape[0]
x32 = icu_yes[awaiting_result_bool & pne_bool].shape[0]
x33 = icu_yes[awaiting_result_bool & pne_bool & intubed_bool].shape[0]
x34 = icu_yes[awaiting_result_bool & pne_bool & intubed_bool & tobacco_bool].shape[0]
x35 = icu_yes[awaiting_result_bool & pne_bool & intubed_bool & copd_bool & tobacco_bool].shape[0]
x36 = icu_yes[awaiting_result_bool & pne_bool & intubed_bool & asthma_bool & tobacco_bool & copd_bool].shape[0]

fig.add_trace(go.Funnel(
    name = 'Positive',
    y = ["Covid Result", "Pneumonia", "Intubed",'Tobacco',"COPD", 'Asthma'],
    x = [x11, x12, x13, x14, x15, x16], textinfo = "value+percent initial"))

fig.add_trace(go.Funnel(
    name = 'Negative', orientation = "h",
    y = ["Covid Result", "Pneumonia", "Intubed", 'Tobacco',"COPD", 'Asthma'],
    x = [x21, x22, x23, x24, x25, x26],
    textposition = "inside", textinfo = "value+percent previous"))

fig.add_trace(go.Funnel(
    name = 'Awaiting Results/Unspecified', orientation = "h",
    y = ["Covid Result", "Pneumonia", "Intubed", 'Tobacco',"COPD", 'Asthma'], x = [x31, x32, x33, x34,x35, x36],
    textposition = "inside",
    textinfo = "value+percent previous"))
fig.update_layout(title="ICU Patients who have most common preconditions ",title_x=0.35,xaxis=dict(title="Number of Patients"),yaxis=dict(title="Preconditions"))
fig.show()
fig = go.Figure()

x21 = icu_yes[negative_result_bool].shape[0]
x22 = icu_yes[negative_result_bool & ~pne_bool].shape[0]
x23 = icu_yes[negative_result_bool & ~pne_bool & ~intubed_bool].shape[0]
x24 = icu_yes[negative_result_bool & ~pne_bool & ~intubed_bool & ~hypertension_bool].shape[0]
x25 = icu_yes[negative_result_bool & ~pne_bool & ~intubed_bool  & ~hypertension_bool & ~diabetes_bool].shape[0]
x26 = icu_yes[negative_result_bool & ~pne_bool & ~intubed_bool  & ~hypertension_bool & ~diabetes_bool & ~obesity_bool].shape[0]

x31 = icu_yes[awaiting_result_bool].shape[0]
x32 = icu_yes[awaiting_result_bool & ~pne_bool].shape[0]
x33 = icu_yes[awaiting_result_bool & ~pne_bool & ~intubed_bool].shape[0]
x34 = icu_yes[awaiting_result_bool & ~pne_bool & ~intubed_bool & ~hypertension_bool].shape[0]
x35 = icu_yes[awaiting_result_bool & ~pne_bool & ~intubed_bool & ~hypertension_bool & ~diabetes_bool ].shape[0]
x36 = icu_yes[awaiting_result_bool & ~pne_bool & ~intubed_bool & ~hypertension_bool & ~diabetes_bool & ~obesity_bool].shape[0]


fig.add_trace(go.Funnel(
    name = 'Negative', orientation = "h",
    y = ["Covid Results - Negative/ Awaiting Results", "No Pneumonia", "No Intubation Required", 'No hypertension','No diabetes',"No obesity"],
    x = [x21, x22, x23, x24, x25, x26],
    textposition = "inside",
    textinfo = "value+percent previous"))

fig.add_trace(go.Funnel(
    name = 'Awaiting Results', orientation = "h",
    y = ["Covid Results - Negative/ Awaiting Results", "No Pneumonia", "No Intubation Required", 'No hypertension','No diabetes',"No obesity"],
    x = [x31, x32, x33, x34,x35, x36],
    textposition = "inside",
    textinfo = "value+percent previous"))
fig.update_layout(title="ICU Patients who did not have usual preconditions ",title_x=0.35,xaxis=dict(title="Number of Patients"),yaxis=dict(title="Preconditions"))

fig.show()
print("Printing the pre-condition of patient numbers: ")
print("Pnemonia patients: ",icu_yes[pne_bool].shape[0])
print("Non Pnemonia patients: ",icu_yes[~pne_bool].shape[0])
print("Percentage of Pnemonia patients: ",(icu_yes[pne_bool].shape[0]/(icu_yes[pne_bool].shape[0] + icu_yes[~pne_bool].shape[0]) * 100))
print("Intubed patients: ",icu_yes[intubed_bool].shape[0])
print("Non Intubed patients: ",icu_yes[~intubed_bool].shape[0])
print("Percentage of Intubed patients: ",(icu_yes[intubed_bool].shape[0]/(icu_yes[intubed_bool].shape[0] + icu_yes[~intubed_bool].shape[0]) * 100))
print("COPD patients: ",icu_yes[copd_bool].shape[0])
print("Non COPD patients: ",icu_yes[~copd_bool].shape[0])
print("Percentage of COPD patients: ",(icu_yes[copd_bool].shape[0]/(icu_yes[copd_bool].shape[0] + icu_yes[~copd_bool].shape[0]) * 100))
print("Asthma patients: ",icu_yes[asthma_bool].shape[0])
print("Non Asthma patients: ",icu_yes[~asthma_bool].shape[0])
print("Percentage of Asthma patients: ",(icu_yes[asthma_bool].shape[0]/(icu_yes[asthma_bool].shape[0] + icu_yes[~asthma_bool].shape[0]) * 100))
print("Tobacco patients: ",icu_yes[tobacco_bool].shape[0])
print("Non Tobacco patients: ",icu_yes[~tobacco_bool].shape[0])
print("Percentage of Tobacco patients: ",(icu_yes[tobacco_bool].shape[0]/(icu_yes[tobacco_bool].shape[0] + icu_yes[~tobacco_bool].shape[0]) * 100))
print("Immuno Supression patients: ",icu_yes[inmsupr_bool].shape[0])
print("Non Immuno Supression patients: ",icu_yes[~inmsupr_bool].shape[0])
print("Percentage of Immuno Supression patients: ",(icu_yes[inmsupr_bool].shape[0]/(icu_yes[inmsupr_bool].shape[0] + icu_yes[~inmsupr_bool].shape[0]) * 100))
print("Hypertension patients: ",icu_yes[hypertension_bool].shape[0])
print("Non Hypertension patients: ",icu_yes[~hypertension_bool].shape[0])
print("Percentage of Hypertension patients: ",(icu_yes[hypertension_bool].shape[0]/(icu_yes[hypertension_bool].shape[0] + icu_yes[~hypertension_bool].shape[0]) * 100))
print("Cardiovascular patients: ",icu_yes[cardiovascular_bool].shape[0])
print("Non Cardiovascular patients: ",icu_yes[~cardiovascular_bool].shape[0])
print("Percentage of Cardiovascular patients: ",(icu_yes[cardiovascular_bool].shape[0]/(icu_yes[cardiovascular_bool].shape[0] + icu_yes[~cardiovascular_bool].shape[0]) * 100))
print("Obesity patients: ",icu_yes[obesity_bool].shape[0])
print("Non Obesity patients: ",icu_yes[~obesity_bool].shape[0])
print("Percentage of Obesity patients: ",(icu_yes[obesity_bool].shape[0]/(icu_yes[obesity_bool].shape[0] + icu_yes[~obesity_bool].shape[0]) * 100))
print("Renal Chronic patients: ",icu_yes[renal_chronic_bool].shape[0])
print("Non Renal Chronic patients: ",icu_yes[~renal_chronic_bool].shape[0])
print("Percentage of Renal Chronic patients: ",(icu_yes[renal_chronic_bool].shape[0]/(icu_yes[renal_chronic_bool].shape[0] + icu_yes[~renal_chronic_bool].shape[0]) * 100))
print("Other disease patients: ",icu_yes[other_disease_bool].shape[0])
print("Non Other disease patients: ",icu_yes[~other_disease_bool].shape[0])
print("Percentage of Other disease_bool patients: ",(icu_yes[other_disease_bool].shape[0]/(icu_yes[other_disease_bool].shape[0] + icu_yes[~other_disease_bool].shape[0]) * 100))
print("Diabetes patients: ",icu_yes[diabetes_bool].shape[0])
print("Non Diabetes patients: ",icu_yes[~diabetes_bool].shape[0])
print("Percentage of Diabetes patients: ",(icu_yes[diabetes_bool].shape[0]/(icu_yes[diabetes_bool].shape[0] + icu_yes[~diabetes_bool].shape[0]) * 100))
tree_df=icu_yes.groupby(['covid_res','pneumonia' ,'intubed','hypertension','diabetes','obesity','tobacco','cardiovascular','renal_chronic'])['id'].agg('count').to_frame(name='count').rename(columns={'id':'count'}).reset_index().sort_values(by='count',ascending=False)
tree_df
tree_df["all"] = "All ICU Patients" # in order to have a single root node
fig = px.treemap(tree_df, path=['all', 'covid_res', 'pneumonia', 'intubed','hypertension','diabetes','obesity'], values='count', width=1600, height=900)#, color='pneumonia', color_continuous_midpoint=100, color_continuous_scale=px.colors.diverging.Portland)
fig.update_layout(treemapcolorway = ["brown", "magenta"])
fig.show()
def categorical_df(disease):
    print(icu_yes.groupby(['covid_res', disease, 'died' ])['id'].agg('count'))
    categorical_df = icu_yes.groupby(['covid_res',disease,'died'])['id'].agg('count').to_frame(name='count').rename(columns={'id':'count'}).reset_index().sort_values(by='count',ascending=False)    
    return categorical_df

disease = categorical_df('inmsupr')
sns.catplot(x="covid_res", y="count", hue="died", col='inmsupr',aspect=.65, kind="swarm", data=disease)
# Create dimensions
other_disease_dim = go.parcats.Dimension(values=icu_yes.other_disease, categoryorder='category descending', label="Any Other Diseases")
contact_dim = go.parcats.Dimension(values=icu_yes.contact_other_covid, categoryorder='category descending',label="Contact Other Covid Patients")
survival_dim = go.parcats.Dimension(values=icu_yes.died, categoryorder='category descending',label="Died" )

# Create parcats trace
color = icu_yes.died;
colorscale = [[0, 'olive'], [0.13, 'violet'], [0.23, 'deeppink'],[0.33, 'firebrick'], [0.66, 'turquoise'], [0.66, 'seagreen'], [1.0, 'powderblue']]#[[0, 'lightsteelblue'], [1, 'firebrick']];

fig = go.Figure(data = [go.Parcats(dimensions=[other_disease_dim, contact_dim, survival_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='color', hoverinfo='count+probability',
        labelfont={'size': 22, 'family': 'Raleway'},
        tickfont={'size': 18, 'family': 'Raleway'},
        arrangement='freeform')])

fig.show()
gender_df=icu_yes.groupby(['sex','covid_res']).agg('count')['id'].to_frame(name='count').reset_index()
gender_male=gender_df.loc[gender_df['sex']=='Male']
gender_female=gender_df.loc[gender_df['sex']=='Female']

male=go.Bar(x=gender_male['covid_res'],y=gender_male['count'],marker=dict(color='brown'),name="male")
female=go.Bar(x=gender_female['covid_res'],y=gender_female['count'],marker=dict(color='orange'),name="female")
data=[male,female]

fig = go.Figure(data)
fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title="Covid Results - Gender (Jan-Jun 2020)",title_x=0.5,xaxis=dict(title="Covid Results"),yaxis=dict(title="Count"),
                   barmode="group")
fig.show()
male_hist_data = icu_yes[icu_yes['sex']=='Male']['age']
female_hist_data = icu_yes[icu_yes['sex']=='Female']['age']
group_labels = ['Male','Female'] 

hist_data = [male_hist_data, female_hist_data]
fig=go.Figure()
fig=ff.create_distplot(hist_data, group_labels,bin_size=12)
fig.update_layout(title_text="Distribution of Age - Gender wise",title_x=0.5)
fig.show()
temp = pd.DataFrame({'age_freq':icu_yes.age_freq.value_counts()})
df = temp
df = df.sort_values(by='age_freq', ascending=True)
data  = go.Data([
            go.Bar(
              y = df.index,
              x = df.age_freq,
              orientation='h'
        )])
layout = go.Layout(
       
        margin=go.layout.Margin(l=300),
        title = "Number of Patients in various age groups"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)
fig = px.parallel_categories(icu_yes, dimensions=['died','covid_res','sex'],
                color="age", color_continuous_scale=px.colors.sequential.Inferno, # Color for Pclass
               labels={'survived_or_not':'Died', 'covid_res':'Covid Result'}) # labeling
fig.update_layout(title="ICU Parallel Categories Diagram ")
fig.show()
print(icu_yes['died'].value_counts())
icu_yes.groupby(['died','covid_res','sex']).agg('count')
headerColor = 'red'
rowEvenColor = 'lightpink'
rowOddColor = 'skyblue'

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b>CONDITIONS</b>','<b>YES</b>','<b>NO</b>','<b>HAVING CONDITION AND DIED</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=16)
  ),
  cells=dict(
    values=[
      ['Pneumonia', 'Intubed', 'COPD', 'Asthma', 'Tobacco', 'Immuno Supression taken','Hypertension','Cardiovascular Disease','Obesity','Renal Chronic Disease','Diabetes','Other Diseases'],
      [icu_yes[pne_bool].shape[0], icu_yes[intubed_bool].shape[0], icu_yes[copd_bool].shape[0], icu_yes[asthma_bool].shape[0], icu_yes[tobacco_bool].shape[0], icu_yes[inmsupr_bool].shape[0], icu_yes[hypertension_bool].shape[0], icu_yes[cardiovascular_bool].shape[0], icu_yes[obesity_bool].shape[0],icu_yes[renal_chronic_bool].shape[0],icu_yes[diabetes_bool].shape[0], icu_yes[other_disease_bool].shape[0]] ,
      [icu_yes[~pne_bool].shape[0], icu_yes[~intubed_bool].shape[0], icu_yes[~copd_bool].shape[0], icu_yes[~asthma_bool].shape[0], icu_yes[~tobacco_bool].shape[0], icu_yes[~inmsupr_bool].shape[0], icu_yes[~hypertension_bool].shape[0],icu_yes[~cardiovascular_bool].shape[0], icu_yes[~obesity_bool].shape[0], icu_yes[~renal_chronic_bool].shape[0], icu_yes[~diabetes_bool].shape[0], icu_yes[~other_disease_bool].shape[0]  ],
      [icu_yes[pne_bool & died_bool].shape[0], icu_yes[intubed_bool & died_bool].shape[0], icu_yes[copd_bool & died_bool].shape[0], icu_yes[asthma_bool & died_bool].shape[0], icu_yes[tobacco_bool & died_bool].shape[0], icu_yes[inmsupr_bool & died_bool].shape[0], icu_yes[hypertension_bool & died_bool].shape[0], icu_yes[cardiovascular_bool & died_bool].shape[0], icu_yes[obesity_bool & died_bool].shape[0],icu_yes[renal_chronic_bool & died_bool].shape[0],icu_yes[diabetes_bool & died_bool].shape[0], icu_yes[other_disease_bool & died_bool].shape[0]   ]   ],
      line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor,rowEvenColor]*5],
    align = ['left', 'center'],
    font = dict(color = 'black', size = 14)
    ))
])

fig.show()
 
print("="* 90)
preg_df = icu_yes[sex_bool & preg_bool]
non_preg_df = icu_yes[sex_bool & ~preg_bool]
print("Women who needed ICU, pregnant is {} ".format(preg_df.shape[0]))
print("Women who needed ICU, non pregnant is {} ".format(icu_yes[sex_bool & ~preg_bool].shape[0]))
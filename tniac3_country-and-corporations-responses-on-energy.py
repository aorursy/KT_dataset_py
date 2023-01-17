import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# to use plotly offline
from plotly.offline import iplot
import plotly.express as px


import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# to ignore warnings
import warnings
warnings.filterwarnings('ignore')

# for Markdown printing
from IPython.display import Markdown, display
def printmd(string, color=None):
    colorstr = "<span style='color:{}'>{}</span>".format(color, string)
    display(Markdown(colorstr))
data_path = '/kaggle/input/cdp-unlocking-climate-solutions'
Cities_Disclosing_2018 = pd.read_csv(data_path+'/Cities/Cities Disclosing/2018_Cities_Disclosing_to_CDP.csv')
Cities_Disclosing_2019 = pd.read_csv(data_path+'/Cities/Cities Disclosing/2019_Cities_Disclosing_to_CDP.csv')
Cities_Disclosing_2020 = pd.read_csv(data_path+'/Cities/Cities Disclosing/2020_Cities_Disclosing_to_CDP.csv')

df_cd = pd.concat([Cities_Disclosing_2018,Cities_Disclosing_2019,Cities_Disclosing_2020],0, ignore_index=True)
Cities_Responses_2018 = pd.read_csv(data_path+'/Cities/Cities Responses/2018_Full_Cities_Dataset.csv')
Cities_Responses_2019 = pd.read_csv(data_path+'/Cities/Cities Responses/2019_Full_Cities_Dataset.csv')
Cities_Responses_2020 = pd.read_csv(data_path+'/Cities/Cities Responses/2020_Full_Cities_Dataset.csv')

df_cr = pd.concat([Cities_Responses_2018,Cities_Responses_2019,Cities_Responses_2020],0, ignore_index=True)
Corporates_Disclosing_to_CDP_Climate_Change_2018 = pd.read_csv(data_path+'/Corporations/Corporations Disclosing/Climate Change/2018_Corporates_Disclosing_to_CDP_Climate_Change.csv')
Corporates_Disclosing_to_CDP_Climate_Change_2019 = pd.read_csv(data_path+'/Corporations/Corporations Disclosing/Climate Change/2019_Corporates_Disclosing_to_CDP_Climate_Change.csv')
Corporates_Disclosing_to_CDP_Climate_Change_2020 = pd.read_csv(data_path+'/Corporations/Corporations Disclosing/Climate Change/2020_Corporates_Disclosing_to_CDP_Climate_Change.csv')

df1 = pd.concat([Corporates_Disclosing_to_CDP_Climate_Change_2018,Corporates_Disclosing_to_CDP_Climate_Change_2019,Corporates_Disclosing_to_CDP_Climate_Change_2020],0, ignore_index=True)
Corporations_Responses_to_Climate_Change_2018 = pd.read_csv(data_path+'/Corporations/Corporations Responses/Climate Change/2018_Full_Climate_Change_Dataset.csv')
Corporations_Responses_to_Climate_Change_2019 = pd.read_csv(data_path+'/Corporations/Corporations Responses/Climate Change/2019_Full_Climate_Change_Dataset.csv')
Corporations_Responses_to_Climate_Change_2020 = pd.read_csv(data_path+'/Corporations/Corporations Responses/Climate Change/2020_Full_Climate_Change_Dataset.csv')

df2 = pd.concat([Corporations_Responses_to_Climate_Change_2018,Corporations_Responses_to_Climate_Change_2019,Corporations_Responses_to_Climate_Change_2020],0, ignore_index=True)
def clean_text_round1(text):
    '''Vietnam, VietNam, Viet nam to Viet Nam'''
    
    text = text.replace("Vietnam",  "Viet Nam")
    text = text.replace("VietNam",  "Viet Nam")
    text = text.replace("Viet nam", "Viet Nam")

    return text

round1 = lambda x: clean_text_round1(x)
df_cd = pd.DataFrame(df_cd.apply(round1))
df_cr = pd.DataFrame(df_cr.apply(round1))
df1 = pd.DataFrame(df1.apply(round1))
df2 = pd.DataFrame(df2.apply(round1))
df_cd.loc[df_cd.Country == 'Viet Nam']
printmd('All sections in cities responses on climate change', color="#255483")
print(df_cr['Section'].unique())
tmp = df_cr.loc[df_cr.Country =='Viet Nam'].Organization.value_counts().to_frame('')
fig = px.bar(tmp,color_discrete_sequence=["#67a9cf"])

fig.update_layout(
    title = dict(text='Number of responses from Viet Nam cities'),
    font  = dict(family="Calibri",size=13,color="RebeccaPurple"),
    xaxis = dict(title='',tickmode='auto'),
    yaxis = dict(title='Count',tickmode='auto'),
#     legend = dict(yanchor="middle",y=0,xanchor="center",x=1),
    width=900, height=500)
hcm_energy = df_cr.loc[(df_cr.Organization =='Ho Chi Minh City') & (df_cr['Section'] =='Energy')]
answers_columns = ['Year Reported to CDP','Column Name','Response Answer',]
for i,q in enumerate(hcm_energy['Question Name'].unique()):
    statement = str(i+1)+ '. ' + q
    printmd(statement, color="#255483")
    display(hcm_energy[answers_columns].loc[hcm_energy['Question Name'] ==q].reset_index(drop=True))
df2_countries = df2[['account_number','organization','response_value']].loc[df2.question_number =='C0.3'].reset_index(drop=True)

print(df2_countries.shape)
df2_countries.head()
df2.module_name.value_counts()
v = []
for i in range(df2_countries.shape[0]):
    if ('Viet Nam' in str(df2_countries.iloc[i,2])): 
        v.append(i)

vn_corps = df2_countries[['account_number','organization']].iloc[v].drop_duplicates()
vn_corps.reset_index(drop=True,inplace=True)

statement = "There are " + str(vn_corps.shape[0]) + ' companies that might have climate change action in Viet Nam'
printmd(statement, color="#255483")
vn_corps.head()
v = []
for i in range(df2.shape[0]):
    tmp = int(df2.iloc[i,0])
    if tmp in vn_corps.account_number.to_list():
        v.append(i)
df2_vn = df2.iloc[v]
df2_vn.head()
df2_vn_en = df2_vn.loc[df2_vn['module_name'] =='C8. Energy']
df2_vn_en.head()
df2_vn_en.loc[(df2_vn_en['question_number']=='C8.1')]['response_value'].value_counts()
df2_vn_en.loc[(df2_vn_en['question_number']=='C8.1')] = df2_vn_en.loc[(df2_vn_en['question_number']=='C8.1')].fillna("Don't know")
df2_vn_en.loc[(df2_vn_en['question_number']=='C8.1')]['response_value'].value_counts()
tmp = df2_vn_en.loc[(df2_vn_en['question_number']=='C8.1') & (df2_vn_en['survey_year'] == 2020)].response_value.value_counts().to_frame('')
fig = px.bar(tmp,color_discrete_sequence=["#67a9cf"])

fig.update_layout(
    title = dict(text='Percentage of total operational spend in the reporting year was on energy, 2020'),
    font  = dict(family="Calibri",size=13,color="RebeccaPurple"),
    xaxis = dict(title='',tickmode='auto'),
    yaxis = dict(title='Number of companies',tickmode='auto'),
    
    width=900, height=600)
tmp = df2_vn_en.loc[df2_vn_en['question_number']=='C8.2']
for i in tmp['row_name'].unique():
    tmp.loc[tmp['row_name'] ==i].response_value.value_counts().iplot(kind='bar',yTitle='Number of companies', 
                                                                     linecolor='black',color='#67a9cf',bargap=0.8,
                                                                     title=i)

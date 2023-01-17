#Import library

import pandas as pd

import numpy as np

from scipy import stats

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



#Set dataframe view options

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', 1000)



#Data cleansing function

def arrange_onehot_var_df(multi_choise_df, onehot_var_list):

    for var in onehot_var_list:

        qestion_no = var[:3]

        rename_df = pd.DataFrame(multi_choise_df[var].value_counts()).reset_index()['index']            

        rename_col = rename_df.values[0]

        multi_choise_df = multi_choise_df.rename(columns={var : qestion_no + rename_col})

        multi_choise_df[qestion_no + rename_col] = multi_choise_df[qestion_no + rename_col].fillna(0)

        multi_choise_df[qestion_no + rename_col] = multi_choise_df[qestion_no + rename_col].apply(lambda x: 0 if x == 0 else 1)

    return multi_choise_df



def question_no_sum(multi_choise_df, quetion_no_list):

    for question_no in quetion_no_list:

        extract_list = [item for item in multi_choise.columns if item.find(question_no) != -1]

        i = 0

        for var in extract_list:

            if i == 0:

                multi_choise_df[question_no + '_sum'] = multi_choise_df[var]

            else:

                multi_choise_df[question_no + '_sum'] = multi_choise_df[question_no + '_sum'] + multi_choise_df[var]

            i = i + 1

    return multi_choise_df



#Make dataframe

multi_choise = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')[1:]



#Drop useless variable

drop_col = [item for item in multi_choise.columns if item.find('_TEXT') != -1]

multi_choise.drop(drop_col, axis=1, inplace=True)



#Data cleansing

onehot_var_list = [item for item in multi_choise.columns if item.find('Part') != -1]

multi_choise = arrange_onehot_var_df(multi_choise_df=multi_choise, onehot_var_list=onehot_var_list)



quetion_no_list = ['Q9','Q12','Q13','Q16','Q17','Q18','Q20','Q21','Q24','Q25',

                   'Q26','Q27','Q28','Q29','Q30','Q31','Q32','Q33','Q34']

multi_choise = question_no_sum(multi_choise_df=multi_choise, quetion_no_list=quetion_no_list)



def Q10_cond(x):

    if  x == '$0-999':

        return '01. $0-999'

    elif x == '1,000-1,999':

        return '02. $1,000-1,999'

    elif x == '2,000-2,999':

        return '03. $2,000-2,999'

    elif x == '3,000-3,999':

        return '04. $3,000-3,999'

    elif x == '4,000-4,999':

        return '05. $4,000-4,999'

    elif x == '5,000-7,499':

        return '06. $5,000-7,499'

    elif x == '7,500-9,999':

        return '07. $7,500-9,999'

    elif x == '10,000-14,999':

        return '08. $10,000-14,999'

    elif x == '15,000-19,999':

        return '09. $15,000-19,999'

    elif x == '20,000-24,999':

        return '10. $20,000-24,999'

    elif x == '25,000-29,999':

        return '11. $25,000-29,999'

    elif x == '30,000-39,999':

        return '12. $30,000-39,999'

    elif x == '40,000-49,999':

        return '13. $40,000-49,999'

    elif x == '50,000-59,999':

        return '14. $50,000-59,999'

    elif x == '60,000-69,999':

        return '15. $60,000-69,999'

    elif x == '70,000-79,999':

        return '16. $70,000-79,999'

    elif x == '80,000-89,999':

        return '17. $80,000-89,999'

    elif x == '90,000-99,999':

        return '18. $90,000-99,999'

    elif x == '100,000-124,999':

        return '19. $100,000-124,999'

    elif x == '125,000-149,999':

        return '20. $125,000-149,999'

    elif x == '150,000-199,999':

        return '21. $150,000-199,999'

    elif x == '200,000-249,999':

        return '22. $200,000-249,999'

    elif x == '250,000-299,999':

        return '23. $250,000-299,999'

    elif x == '300,000-500,000':

        return '24. $300,000-500,000'

    elif x == '> $500,000':

        return '25. $500,000 and over'



multi_choise['Q10'] = multi_choise['Q10'].apply(Q10_cond)



def Q10_earn_cond(x):

    if x == '01. $0-999':

        return 'Not earn'

    elif x == '02. $1,000-1,999':

        return 'Not earn'

    elif x == '03. $2,000-2,999':

        return 'Not earn'

    elif x == '04. $3,000-3,999':

        return 'Not earn'

    elif x == '05. $4,000-4,999':

        return 'Not earn'

    elif x == '06. $5,000-7,499':

        return 'Not earn'

    elif x == '07. $7,500-9,999':

        return 'Not earn'

    elif x == '08. $10,000-14,999':

        return 'Not earn'

    elif x == '09. $15,000-19,999':

        return 'Not earn'

    elif x == '10. $20,000-24,999':

        return 'Not earn'

    elif x == '11. $25,000-29,999':

        return 'Not earn'

    elif x == '12. $30,000-39,999':

        return 'Not earn'

    elif x == '13. $40,000-49,999':

        return 'Not earn'

    elif x == '14. $50,000-59,999':

        return 'Earn'

    elif x == '15. $60,000-69,999':

        return 'Earn'

    elif x == '16. $70,000-79,999':

        return 'Earn'

    elif x == '17. $80,000-89,999':

        return 'Earn'

    elif x == '18. $90,000-99,999':

        return 'Earn'

    elif x == '19. $100,000-124,999':

        return 'Earn'

    elif x == '20. $125,000-149,999':

        return 'Earn'

    elif x == '21. $150,000-199,999':

        return 'Earn'

    elif x == '22. $200,000-249,999':

        return 'Earn'

    elif x == '23. $250,000-299,999':

        return 'Earn'

    elif x == '24. $300,000-500,000':

        return 'Earn'

    elif x == '25. $500,000 and over':

        return 'Earn'



multi_choise['Q10_earn_flg'] = multi_choise['Q10'].apply(Q10_earn_cond)



#Fill Nan

multi_choise = multi_choise.fillna('No Answer')



#Extract Earn or not

multi_choise = multi_choise[multi_choise['Q10_earn_flg'] != 'No Answer']
#Calc Earn Rate in occupations

def calc_earn_rate(df, occupation_list):

    i = 0

    for occupation in occupation_list:

        gb_df = df[(df['Q5'] == occupation)]

        gb_df = pd.DataFrame(gb_df.groupby('Q10_earn_flg').count()['Q5']).reset_index()

        rate = round(gb_df['Q5'][0] / gb_df['Q5'].sum() * 100, 1)

        if i == 0:

            rate_df = pd.DataFrame([[rate]], columns=['rate'])

            rate_df['Earn'] = gb_df['Q5'][0]

            rate_df['Not Earn'] = gb_df['Q5'][1]

            rate_df['sum'] = gb_df['Q5'].sum()

            rate_df['occupation'] = occupation

            i = i + 1

        else:

            tmp_rate_df = pd.DataFrame([[rate]], columns=['rate'])

            tmp_rate_df['Earn'] = gb_df['Q5'][0]

            tmp_rate_df['Not Earn'] = gb_df['Q5'][1]

            tmp_rate_df['sum'] = gb_df['Q5'].sum()

            tmp_rate_df['occupation'] = occupation

            rate_df = rate_df.append(tmp_rate_df)

    return rate_df



occupation_list = ['Data Scientist', 'Software Engineer', 'Data Analyst', 'Research Scientist', 'Business Analyst', 

                   'Product/Project Manager', 'Data Engineer', 'Statistician', 'DBA/Database Engineer']



occupation_rate_df = calc_earn_rate(df=multi_choise, occupation_list=occupation_list)



#Add occupation All

rate = round(occupation_rate_df['Earn'].sum() / occupation_rate_df['sum'].sum() * 100, 1)

tmp_rate_df = pd.DataFrame([[rate]], columns=['rate'])

tmp_rate_df['Earn'] = occupation_rate_df['Earn'].sum()

tmp_rate_df['Not Earn'] = occupation_rate_df['Not Earn'].sum()

tmp_rate_df['sum'] = occupation_rate_df['sum'].sum()

tmp_rate_df['occupation'] = 'ALL'

occupation_rate_df = occupation_rate_df.append(tmp_rate_df)



#Sort

occupation_rate_df = occupation_rate_df.sort_values('rate')



#Make graph

colors=['blue',] * 10

colors[8]='crimson'



x=occupation_rate_df['rate'].values

y=occupation_rate_df['occupation'].values



fig = go.Figure(data=[go.Bar(x=x,y=y,text=x,textposition='auto',orientation='h',marker_color=colors)])

fig.update_layout(title='"Earn" rate by occupation')

fig.show()
#Make graph function

def make_onehot_var_graph(df, var, drop_list, title, height):

    qvar_list = [item for item in df.columns if item.find(var) != -1]

    qvar_list.append('Q10_earn_flg')



    gb_df = ds[qvar_list]

    gb_df = gb_df.groupby('Q10_earn_flg').sum()



    df_col = []

    for i in range(len(qvar_list)-1):

        tmp_df_col = qvar_list[i][3:]

        df_col.append(tmp_df_col)



    gb_df.columns = df_col

    gb_df.drop(drop_list, axis=1, inplace=True)

    gb_df = gb_df.T.reset_index()

    gb_df['Earn_rate'] = round(gb_df['Earn'] / earn * 100, 1)

    gb_df['Not earn_rate'] = round(gb_df['Not earn'] / notearn * 100, 1)



    if var == 'Q9':

        def Q9_cond(x):

            if  x == 'Do research that advances the state of the art of machine learning':

                return 'Research that advances the cutting-edge ML'

            elif x == 'Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data':

                return 'Build the data infrastructure'

            elif x == 'None of these activities are an important part of my role at work':

                return 'I do not play an important role in data science'

            elif x == 'Build prototypes to explore applying machine learning to new area':

                return 'Build prototypes to explore applying ML to new area'

            elif x == 'Analyze and understand data to influence product or business decisions':

                return 'Analyze data to influence business decisions'

            elif x == 'Build and/or run a machine learning service that operationally improves my product or workflows':

                return 'Build ML service that operationally improves our products'

            elif x == 'Other':

                return 'Other'



        gb_df['index'] = gb_df['index'].apply(Q9_cond)



    if var == 'Q12':

        def Q12_cond(x):

            if  x == 'None':

                return 'None'

            elif x == 'Other':

                return 'Other'

            elif x == 'Twitter (data science influencers)':

                return 'Twitter'

            elif x == 'Hacker News (https://news.ycombinator.com/)':

                return 'Hacker News'

            elif x == 'Reddit (r/machinelearning, r/datascience, etc)':

                return 'Reddit'

            elif x == 'Kaggle (forums, blog, social media, etc)':

                return 'Kaggle'

            elif x == 'Course Forums (forums.fast.ai, etc)':

                return 'Course Forums'

            elif x == 'YouTube (Cloud AI Adventures, Siraj Raval, etc)':

                return 'YouTube'

            elif x == 'Podcasts (Chai Time Data Science, Linear Digressions, etc)':

                return 'Podcasts'

            elif x == 'Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)':

                return 'Blogs'

            elif x == 'Journal Publications (traditional publications, preprint journals, etc)':

                return 'Journal Publications'

            elif x == 'Slack Communities (ods.ai, kagglenoobs, etc)':

                return 'Slack Communities'

        

        gb_df['index'] = gb_df['index'].apply(Q12_cond)



    if var == 'Q26':

        def Q26_cond(x):

            if  x == 'None':

                return 'None'

            elif x == 'Object detection methods (YOLOv3, RetinaNet, etc)':

                return 'Object detection methods (YOLOv3, RetinaNet, etc)'

            elif x == 'Other':

                return 'Other'

            elif x == 'General purpose image/video tools (PIL, cv2, skimage, etc)':

                return 'General purpose image/video tools (PIL, cv2, etc)'

            elif x == 'Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)':

                return 'Image classification and other general purpose networks (VGG, ResNet, etc)'

            elif x == 'Build and/or run a machine learning service that operationally improves my product or workflows':

                return 'Build ML service that operationally improves our products'

            elif x == 'Image segmentation methods (U-Net, Mask R-CNN, etc)':

                return 'Image segmentation methods (U-Net, Mask R-CNN, etc)'



        gb_df['index'] = gb_df['index'].apply(Q26_cond)

    

    gb_df['dif_Earn_Notearn'] = gb_df['Earn_rate'] - gb_df['Not earn_rate']

    gb_df = gb_df.sort_values('dif_Earn_Notearn', ascending=False)



    X=list(gb_df['index'].values)

    Y1=gb_df['Earn_rate'].values

    Y2=gb_df['Not earn_rate'].values



    fig = go.Figure(data=[

        go.Bar(name='Earn', x=X, y=Y1, text=Y1, textposition='auto'),

        go.Bar(name='Not Earn', x=X, y=Y2, text=Y2, textposition='auto')

    ])



    fig.update_layout(title=title, barmode='group', height=height, font_size=10, xaxis_tickangle=45)

    fig.show()



def make_onehot_var_sum_avg_graph(df, var, avg_var, title):

    qvar_list = [item for item in df.columns if item.find(var) != -1]

    qvar_list.append('Q10_earn_flg')



    gb_df = ds[qvar_list]

    

    x = gb_df['Q10_earn_flg'].values

    y = gb_df[var+avg_var].values



    earn_df = gb_df[gb_df['Q10_earn_flg'] == 'Earn']

    nearn_df = gb_df[gb_df['Q10_earn_flg'] == 'Not earn']

    

    p_eran = earn_df[var+avg_var].values

    p_nearn = nearn_df[var+avg_var].values

    p_earn_nearn = stats.ttest_ind(p_eran, p_nearn, equal_var=False)

    pvalue = round(p_earn_nearn[1],3)

    name = title+'(p-value='+str(pvalue)+')'



    earn_mean = round(earn_df[var+avg_var].mean(),2)

    nearn_mean = round(nearn_df[var+avg_var].mean(),2)

    

    fig = go.Figure(data=[

        go.Bar(name='Earn', x=["Earn or Not earn"], y=[earn_mean], text=earn_mean, textposition='auto'),

        go.Bar(name='Not earn', x=["Earn or Not earn"], y=[nearn_mean], text=nearn_mean, textposition='auto')

    ])

    fig.update_layout(title=name, font_size=10, height=300, barmode='group')

    fig.show()



#Make Data Scientist data frame

ds = multi_choise[multi_choise['Q5'] == 'Data Scientist']

notearn = ds['Q10_earn_flg'].value_counts()[0]

earn = ds['Q10_earn_flg'].value_counts()[1]
make_onehot_var_graph(df=ds, var='Q9', drop_list='sum', title='Response rate', height=450)

make_onehot_var_sum_avg_graph(df=ds, var='Q9', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q12', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q12', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q16', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q16', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q18', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q18', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q20', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q20', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q24', drop_list='_sum', title='Response rate', height=450)

make_onehot_var_sum_avg_graph(df=ds, var='Q24', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q28', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q28', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q29', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q29', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q31', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q31', avg_var='_sum', title='Number of checks for questioning options')
make_onehot_var_graph(df=ds, var='Q34', drop_list='_sum', title='Response rate', height=300)

make_onehot_var_sum_avg_graph(df=ds, var='Q34', avg_var='_sum', title='Number of checks for questioning options')
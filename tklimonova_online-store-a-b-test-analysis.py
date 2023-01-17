#let's import all the needed libraries for the analysis

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
#upload grossery website data

grossery_web_data_orig = pd.read_csv('../input/grocery-website-data-for-ab-test/grocerywebsiteabtestdata.csv')

grossery_web_data_orig.head()
#grouping to make one row per IP address

grossery_web_data = grossery_web_data_orig.groupby(['IP Address', 'LoggedInFlag', 'ServerID'])['VisitPageFlag'].sum().reset_index(name='sum_VisitPageFlag')
#checking if there is IP address with more than 1 visit

grossery_web_data['visitFlag'] = grossery_web_data['sum_VisitPageFlag'].apply(lambda x: 1 if x !=0 else 0)

grossery_web_data.head()
#creating groups for control and treatment

grossery_web_data['group'] = grossery_web_data.ServerID.map({1:'Treatment', 2:'Control', 3:'Control'})
grossery_web_data.dtypes
#removing all records where the LoggedInFlag=1, so it filters out all the users with accounts

grossery_web_data = grossery_web_data[grossery_web_data['LoggedInFlag'] != 1]

grossery_web_data
treatment = grossery_web_data[grossery_web_data['group']=='Treatment']

control = grossery_web_data[grossery_web_data['group']=='Control']



ttest_ind(treatment['visitFlag'], control['visitFlag'], equal_var = False)
#let's calculate the differences in means

grossery_web_data_diff_mean = grossery_web_data.groupby(['group', 'visitFlag'])['group'].count().reset_index(name='Count')

grossery_web_data_diff_mean
grossery_web_data.groupby('group').visitFlag.mean()
#crosstab by groups

groupped = pd.crosstab(grossery_web_data_diff_mean['group'], grossery_web_data_diff_mean['visitFlag'], values=grossery_web_data_diff_mean['Count'], aggfunc=np.sum, margins=True)

groupped
#Percentage row

100*groupped.div(groupped['All'], axis=0)
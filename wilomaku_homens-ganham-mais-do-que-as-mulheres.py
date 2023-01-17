import pandas as pd

import numpy as np

import seaborn as sns

import missingno as msno

import matplotlib.pyplot as plt

import scipy.stats as st

import matplotlib.patches as mpatches



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_ds = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')

print(df_ds.shape)

df_ds.head()
sns.countplot(df_ds["('P2', 'gender')"])

plt.show()
col_search = "('P8', 'degreee_level')"



df_agg = df_ds.groupby([col_search,"('P2', 'gender')"]).agg({col_search:'count',})

df_agg = df_agg/df_agg.groupby(level=(1)).sum()



df_agg.reset_index(level=1, inplace=True)



plt.figure(figsize=(16,6))



colors = ['orange' if s == 'Feminino' else 'lightblue' for s in df_agg["('P2', 'gender')"]]

plt.bar(np.arange(len(df_agg.index)),df_agg[col_search],color=colors)

plt.xticks(np.arange(len(df_agg.index)),df_agg.index,rotation=90)



red_patch = mpatches.Patch(color='orange', label='Feminino')

blue_patch = mpatches.Patch(color='lightblue', label='Masculino')



plt.legend(handles=[red_patch, blue_patch])

plt.show()
df_ds["('P10', 'job_situation')"].value_counts()*100/df_ds.size
col_search = "('P10', 'job_situation')"



df_agg = df_ds.groupby([col_search,"('P2', 'gender')"]).agg({col_search:'count',})

df_agg = df_agg/df_agg.groupby(level=(1)).sum()



df_agg.reset_index(level=1, inplace=True)



plt.figure(figsize=(16,6))



colors = ['orange' if s == 'Feminino' else 'lightblue' for s in df_agg["('P2', 'gender')"]]

plt.bar(np.arange(len(df_agg.index)),df_agg[col_search],color=colors)

plt.xticks(np.arange(len(df_agg.index)),df_agg.index,rotation=90)



red_patch = mpatches.Patch(color='orange', label='Feminino')

blue_patch = mpatches.Patch(color='lightblue', label='Masculino')



plt.legend(handles=[red_patch, blue_patch])

plt.show()
def transform_string(st):

    

    limits = [float(s[:-4].replace('.','')) for s in st.split() if s[-4:]=='/mês']

    if len(limits) == 2:

        value = np.mean(np.array(limits))

    elif limits[0]>10000:

        value = limits[0]

    else:

        value = limits[0]/2

    return value



df_ds["('P16', 'salary_range')"].fillna('0.0/mês',inplace=True)



df_ds['salary'] = df_ds.apply(lambda row: transform_string(row["('P16', 'salary_range')"]),axis=1)



df_ds[["('P16', 'salary_range')",'salary']].head()
sns.distplot(df_ds["salary"],bins=20)

plt.show()
sns.distplot(df_ds.loc[df_ds["('P2', 'gender')"]=='Masculino',["salary"]],bins=20,label='Masculino')

sns.distplot(df_ds.loc[df_ds["('P2', 'gender')"]=='Feminino',["salary"]],bins=20,label='Feminino')

plt.legend(loc=0)

plt.show()
col_search = "('P10', 'job_situation')"



medianprops = dict(linestyle='-', linewidth=4)

bp_dict = df_ds.boxplot(column='salary',by=[col_search,"('P2', 'gender')"],rot=90,

                        figsize=(20,8),fontsize=16,grid=True,return_type='both',patch_artist = True,

                        medianprops=medianprops)



df_agg = df_ds.groupby([col_search,"('P2', 'gender')"]).agg({col_search:'count',})

df_agg = df_agg/df_agg.groupby(level=(1)).sum()



df_agg.reset_index(level=1, inplace=True)



colors = ['orange' if s == 'Feminino' else 'lightblue' for s in df_agg["('P2', 'gender')"]]



for row_key, (ax,row) in bp_dict.iteritems():

    ax.set_xlabel('')

    for i,box in enumerate(row['boxes']):

        box.set_facecolor(colors[i])



red_patch = mpatches.Patch(color='orange', label='Feminino')

blue_patch = mpatches.Patch(color='lightblue', label='Masculino')



plt.legend(handles=[red_patch, blue_patch])

plt.title('Diferenças de salário por tipo de trabalho entre homens e mulheres')

plt.show()
df_stats = df_ds.groupby("('P2', 'gender')").agg({'salary':['mean','std','count']})

mean_fem, std_fem, n_fem = df_stats.loc['Feminino']

mean_mas, std_mas, n_mas = df_stats.loc['Masculino']



media_dif = mean_mas - mean_fem



beta = 0.975 #Bicaudal

z_norm = st.norm.ppf(beta)



std_dif = np.sqrt((std_mas**2)/n_mas+(std_fem**2)/n_fem)

d = z_norm*std_dif



z = media_dif/std_dif

p_value = 1 - st.norm.cdf(z)



df_st = pd.DataFrame([['global',n_fem,n_mas,media_dif,std_dif,(media_dif-d),(media_dif+d),z,p_value],],

                     columns=['job','n feminino','n masculino','mean_dif','std_dif','inter_inf','inter_sup','z','p_value'])

df_st
col_search = "('P10', 'job_situation')"



for job_type in ['Empregado (CTL)', 'Estagiário', 'Freelancer','Empreendedor ou Empregado (CNPJ)',

                 'Servidor Público',]:





    df_stats = df_ds.loc[df_ds[col_search]==job_type].groupby("('P2', 'gender')").agg({'salary':['mean','std','count']})

    mean_fem, std_fem, n_fem = df_stats.loc['Feminino']

    mean_mas, std_mas, n_mas = df_stats.loc['Masculino']



    media_dif = mean_mas - mean_fem



    std_dif = np.sqrt((std_mas**2)/n_mas+(std_fem**2)/n_fem)

    d = z_norm*std_dif



    z = media_dif/std_dif

    p_value = 1 - st.norm.cdf(z)



    df_st = df_st.append(pd.DataFrame([[job_type,n_fem,n_mas,media_dif,std_dif,(media_dif-d),(media_dif+d),z,p_value],],

                         columns=['job','n feminino','n masculino','mean_dif','std_dif','inter_inf','inter_sup','z','p_value']),

                         ignore_index=True)

df_st
col_search = "('P17', 'time_experience_data_science')"



df_agg = df_ds.groupby([col_search,"('P2', 'gender')"]).agg({col_search:'count',})

df_agg = df_agg/df_agg.groupby(level=(1)).sum()



df_agg.reset_index(level=1, inplace=True)



plt.figure(figsize=(16,6))



colors = ['orange' if s == 'Feminino' else 'lightblue' for s in df_agg["('P2', 'gender')"]]

plt.bar(np.arange(len(df_agg.index)),df_agg[col_search],color=colors)

plt.xticks(np.arange(len(df_agg.index)),df_agg.index,rotation=90)



red_patch = mpatches.Patch(color='orange', label='Feminino')

blue_patch = mpatches.Patch(color='lightblue', label='Masculino')



plt.legend(handles=[red_patch, blue_patch])

plt.show()
medianprops = dict(linestyle='-', linewidth=4)

bp_dict = df_ds.boxplot(column='salary',by=[col_search,"('P2', 'gender')"],rot=90,

                        figsize=(20,8),fontsize=16,grid=True,return_type='both',patch_artist = True,

                        medianprops=medianprops)



df_agg = df_ds.groupby([col_search,"('P2', 'gender')"]).agg({col_search:'count',})

df_agg = df_agg/df_agg.groupby(level=(1)).sum()



df_agg.reset_index(level=1, inplace=True)



colors = ['orange' if s == 'Feminino' else 'lightblue' for s in df_agg["('P2', 'gender')"]]



for row_key, (ax,row) in bp_dict.iteritems():

    ax.set_xlabel('')

    for i,box in enumerate(row['boxes']):

        box.set_facecolor(colors[i])



red_patch = mpatches.Patch(color='orange', label='Feminino')

blue_patch = mpatches.Patch(color='lightblue', label='Masculino')



plt.legend(handles=[red_patch, blue_patch])

plt.title('Diferenças de salário por tipo de trabalho entre homens e mulheres')

plt.show()
df_st = pd.DataFrame([], columns=['job','n feminino','n masculino','mean_dif','std_dif','inter_inf','inter_sup','z','p_value'])



for exp_type in ['Não tenho experiência na área de dados','Menos de 1 ano','de 1 a 2 anos','de 2 a 3 anos','de 4 a 5 anos','de 6 a 10 anos','Mais de 10 anos',]:





    df_stats = df_ds.loc[df_ds[col_search]==exp_type].groupby("('P2', 'gender')").agg({'salary':['mean','std','count']})

    mean_fem, std_fem, n_fem = df_stats.loc['Feminino']

    mean_mas, std_mas, n_mas = df_stats.loc['Masculino']



    media_dif = mean_mas - mean_fem



    std_dif = np.sqrt((std_mas**2)/n_mas+(std_fem**2)/n_fem)

    d = z_norm*std_dif



    z = media_dif/std_dif

    p_value = 1 - st.norm.cdf(z)



    df_st = df_st.append(pd.DataFrame([[exp_type,n_fem,n_mas,media_dif,std_dif,(media_dif-d),(media_dif+d),z,p_value],],

                         columns=['job','n feminino','n masculino','mean_dif','std_dif','inter_inf','inter_sup','z','p_value']),

                         ignore_index=True)

df_st
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set(style='white',palette='deep')

width = 0.35

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%matplotlib inline

#Function

def autolabel_without_pct(rects,ax): #autolabel

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy = (rect.get_x() + rect.get_width()/2, height),

                    xytext= (0,3),

                    textcoords="offset points",

                    ha='center', va='bottom')
#Importing Dataset

df = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')

df.info()
#Feature Engineering

df_feature = df.copy()

df_feature.columns

np.unique(df_feature['category'])
#Null Values

df_feature.isnull().sum()

percentage_null = (df_feature.isnull().sum()/len(df_feature))*100

percentage_null = pd.DataFrame(percentage_null, columns= ['% Null Values'])

percentage_null
#Which year had more Categories?

from collections import Counter

years = Counter(df_feature['year_award'])

labels = [list(years.keys())[i] for i in np.arange(0,len(years))]

years_category_count = [list(years.values())[i] for i in np.arange(0,len(years))]

ind = np.arange(len(years_category_count))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

for i in np.arange(0,len(years_category_count)):

    rects = ax.barh(labels[i],years_category_count[i], width,edgecolor='black')

    plt.tight_layout()

ax.set_ylabel('Year Award', fontsize=10)

ax.set_xlabel('Amount of Categories', fontsize=10)

ax.grid(b=True, which='major', linestyle='--')

ax.set_title('Amount of Categories \n by Years', fontsize=10)
#Top 10 Nominee and Film Winners Golden Global Award (Plotting)

df_win = df_feature[df_feature['win']==True]

df_win.columns

film_win = Counter(df_win['film'])

nominee_win = Counter(df_win['nominee'])

film_win = {k: v for k, v in sorted(film_win.items(), key=lambda item: item[1], reverse=True)}

nominee_win = {k: v for k, v in sorted(nominee_win.items(), key=lambda item: item[1], reverse=True)}



film_win_values = [list(film_win.values())[i] for i in np.arange(0,len(film_win))]

film_win_labels = [list(film_win.keys())[i] for i in np.arange(0,len(film_win))]

nominee_win_values = [list(nominee_win.values())[i] for i in np.arange(0,len(nominee_win))]

nominee_win_labels = [list(nominee_win.keys())[i] for i in np.arange(0,len(nominee_win))]



film_win_labels.remove(np.nan)

film_win_values.remove(425)



ind=np.arange(len(film_win_values[0:11]))

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

for i in np.arange(len(film_win_values[0:11])):

    rects1 = ax1.bar(film_win_labels[i],film_win_values[i], width=width, edgecolor='black', align='center')

    rects2 = ax2.bar(nominee_win_labels[i],nominee_win_values[i],width=width, edgecolor='black', align='center')

    autolabel_without_pct(rects1,ax1)

    autolabel_without_pct(rects2, ax2)

    

ax1.set_xticks(ind)

ax1.set_xlabel('Movie / Serie / Documentary', fontsize=10)

ax1.set_xticklabels(film_win_labels, fontsize=10)

ax1.set_ylabel('Number of Prizes', fontsize=10)

ax1.set_title('Top 10 Film Golden Globe Award Winners', fontsize=10)

ax1.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax1.grid(b=True, which='major', linestyle='--')

ax1.set_ylim(0,10)



ax2.set_xticks(ind)

ax2.set_xlabel('Nominee', fontsize=10)

ax2.set_xticklabels(nominee_win_labels, fontsize=10)

ax2.set_ylabel('Number of Prizes', fontsize=10)

ax2.set_title('Top 10 nominee Golden Globe Award Winners', fontsize=10)

ax2.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax2.grid(b=True, which='major', linestyle='--')

plt.tight_layout()

ax2.set_ylim(0,10)
#Top 10 Nominee and Film Winners Golden Global Award (Pandas)

film = pd.DataFrame()

film = [df_win[df_win['film'] == film_win_labels[i]] for i in np.arange(0,11)]

film_final = pd.DataFrame([film[x].values[i] for x in np.arange(len(film)) for i in np.arange(len(film[x]))], columns=df_win.columns)



nominee = pd.DataFrame()

nominee = [df_win[df_win['nominee'] == nominee_win_labels[i]] for i in np.arange(0,11)]

nominiee_final = pd.DataFrame([nominee[x].values[i] for x in np.arange(len(nominee)) for i in np.arange(len(nominee[x]))], columns=df_win.columns)

nominiee_final
#Top 10 Nominee and Film indication Golden Global Award (Plotting)

df_feature.columns

film_indication = Counter(df_feature['film'])

nominee_indication = Counter(df_feature['nominee'])

film_indication = {k: v for k, v in sorted(film_indication.items(), key=lambda item: item[1], reverse=True)}

nominee_indication = {k: v for k, v in sorted(nominee_indication.items(), key=lambda item: item[1], reverse=True)}



film_indication_values = [list(film_indication.values())[i] for i in np.arange(0,len(film_indication))]

film_indication_labels = [list(film_indication.keys())[i] for i in np.arange(0,len(film_indication))]

nominee_indication_values = [list(nominee_indication.values())[i] for i in np.arange(0,len(nominee_indication))]

nominee_indication_labels = [list(nominee_indication.keys())[i] for i in np.arange(0,len(nominee_indication))]



film_indication_labels.remove(np.nan)

film_indication_values.remove(1800)



ind=np.arange(len(film_indication_values[0:11]))

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,1,1)

ax2 = fig.add_subplot(2,1,2)

for i in np.arange(len(film_indication_values[0:11])):

    rects1 = ax1.bar(film_indication_labels[i],film_indication_values[i], width=width, edgecolor='black', align='center')

    rects2 = ax2.bar(nominee_indication_labels[i],nominee_indication_values[i],width=width, edgecolor='black', align='center')

    autolabel_without_pct(rects1,ax1)

    autolabel_without_pct(rects2, ax2)



ax1.set_xticks(ind)

ax1.set_xlabel('Movie / Serie / Documentary', fontsize=10)

ax1.set_xticklabels(film_indication_labels, fontsize=10)

ax1.set_ylabel('Number of Indications', fontsize=10)

ax1.set_title('Top 10 Film Golden Globe Award Indications', fontsize=10)

ax1.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax1.grid(b=True, which='major', linestyle='--')

ax1.set_ylim(0,40)



ax2.set_xticks(ind)

ax2.set_xlabel('Nominee', fontsize=10)

ax2.set_xticklabels(nominee_indication_labels, fontsize=10)

ax2.set_ylabel('Number of Indications', fontsize=10)

ax2.set_title('Top 10 Nominee Golden Globe Award Indications', fontsize=10)

ax2.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax2.grid(b=True, which='major', linestyle='--')

plt.tight_layout()

ax2.set_ylim(0,40)
#Top 10 Nominee and Film Indications Golden Global Award (Pandas)

film_indication = pd.DataFrame()

film_indication = [df_feature[df_feature['film'] == film_indication_labels[i]] for i in np.arange(0,11)]

film_final_indication = pd.DataFrame([film_indication[x].values[i] for x in np.arange(len(film_indication)) for i in np.arange(len(film_indication[x]))], columns=df_win.columns)

film_final_indication
#Top 10 Nominee and Film Indications Golden Global Award (Pandas)

nominee_indication = pd.DataFrame()

nominee_indication = [df_feature[df_feature['nominee'] == nominee_indication_labels[i]] for i in np.arange(0,11)]

nominiee_final_indication = pd.DataFrame([nominee_indication[x].values[i] for x in np.arange(len(nominee_indication)) for i in np.arange(len(nominee_indication[x]))], columns=df_win.columns)

nominiee_final_indication
#which actor/actress has won more awards? (Plotting)

import re

df_feature.columns

df_actor_actress = df_feature.loc[df_feature['category'].str.contains('actor|actress',flags=re.IGNORECASE, regex=True)]

df_actor_actress = df_actor_actress[df_actor_actress['win']==True]



actor_actress_win = Counter(df_actor_actress['nominee'])

actor_actress_win = {k: v for k, v in sorted(actor_actress_win.items(), key=lambda item: item[1], reverse=True)}



actor_actress_win_values = [list(actor_actress_win.values())[i] for i in np.arange(0,len(actor_actress_win))]

actor_actress_win_labels = [list(actor_actress_win.keys())[i] for i in np.arange(0,len(actor_actress_win))]



ind = np.arange(len(actor_actress_win_labels[:11]))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

for i in np.arange(0,11):

    rects = ax.bar(actor_actress_win_labels[i], actor_actress_win_values[i], width=width, edgecolor='k', align='center')

    autolabel_without_pct(rects,ax)

ax.set_xticks(ind)

ax.set_xlabel('Actor / Actress', fontsize=10)

ax.set_xticklabels(actor_actress_win_labels,fontsize=10)

ax.set_ylabel('Number of Prizes', fontsize=10)

ax.set_ylim(0,10)

ax.set_title('Top 10 Actor / Actress Golden Globe Award Winner', fontsize=10)

ax.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax.grid(b=True, which='major', linestyle='--')

plt.tight_layout()
#which actor/actress has won more awards? (Pandas)

actor_actress = pd.DataFrame()

actor_actress = [df_actor_actress[df_actor_actress['nominee'] == actor_actress_win_labels[i]] for i in np.arange(0,11)]

actor_actress_final = pd.DataFrame([actor_actress[x].values[i] for x in np.arange(len(actor_actress)) for i in np.arange(len(actor_actress[x]))], columns=df_actor_actress.columns)

actor_actress_final
#which actor/actress has more indication? (Plotting)

df_actor_actress_ind = df_feature.loc[df_feature['category'].str.contains('actor|actress',flags=re.IGNORECASE, regex=True)]

actor_actress_ind = Counter(df_actor_actress_ind['nominee'])

actor_actress_ind = {k: v for k, v in sorted(actor_actress_ind.items(), key=lambda item: item[1], reverse=True)}



actor_actress_ind_values = [list(actor_actress_ind.values())[i] for i in np.arange(0,len(actor_actress_ind))]

actor_actress_ind_labels = [list(actor_actress_ind.keys())[i] for i in np.arange(0,len(actor_actress_ind))]



ind = np.arange(len(actor_actress_ind_labels[:11]))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

for i in np.arange(0,11):

    rects = ax.bar(actor_actress_ind_labels[i], actor_actress_ind_values[i], width=width, edgecolor='k', align='center')

    autolabel_without_pct(rects,ax)

ax.set_xticks(ind)

ax.set_xlabel('Actor / Actress', fontsize=10)

ax.set_xticklabels(actor_actress_win_labels,fontsize=10)

ax.set_ylabel('Number of Indications', fontsize=10)

ax.set_ylim(0,35)

ax.set_title('Top 10 Actor / Actress Golden Globe Award Indications', fontsize=10)

ax.tick_params(axis='x', labelsize=10, labelcolor='k', labelrotation=90)

ax.grid(b=True, which='major', linestyle='--')

plt.tight_layout()
#which actor/actress has more indication? (Pandas)

actor_actress_ind = pd.DataFrame()

actor_actress_ind = [df_actor_actress_ind[df_actor_actress_ind['nominee'] == actor_actress_ind_labels[i]] for i in np.arange(0,11)]

actor_actress_ind_final = pd.DataFrame([actor_actress_ind[x].values[i] for x in np.arange(len(actor_actress_ind)) for i in np.arange(len(actor_actress_ind[x]))], columns=df_actor_actress_ind.columns)

actor_actress_ind_final
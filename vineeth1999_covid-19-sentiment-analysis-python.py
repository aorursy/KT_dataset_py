import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from PIL import Image

from datetime import time,date

import nltk

import spacy

import re
data = [[[5,164,192],[133,206,218],[210,167,216]],[[166,123,197],[187,28,139],[220,38,110]],]

fig = plt.figure(figsize=(5, 5))

fig.patch.set_visible(False) 

img = plt.imshow(data,interpolation='nearest')

img.set_cmap('hot')

plt.axis('off')
data = pd.read_csv('../input/coronavirus-2019ncov/covid-19-all.csv')

data.rename(columns={"Country":"Country/Region","State":"State/Province"},inplace=True)

tweets = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')

#Inspect Data

data.head(5)
#Inspect Tweet

tweets.head(2)
data[['Confirmed','Recovered','Deaths']] = data[['Confirmed','Recovered','Deaths']].fillna(0)

data_new = pd.melt(data[['Date','Confirmed','Recovered','Deaths']],id_vars=['Date'],value_vars=['Confirmed','Recovered','Deaths'],var_name='group_var',value_name='Cases')

data_new['Date'] = pd.to_datetime(data_new['Date'])

data['Date'] = pd.to_datetime(data['Date'])

dates = data['Date'].unique()

new_df = pd.DataFrame(index = pd.date_range(dates.min(), dates.max()),columns=['Confirmed','Recovered','Deaths'])

new_df[['Confirmed','Recovered','Deaths']] = new_df.apply(lambda x:data.loc[(data['Date'] == x.name),['Confirmed','Recovered','Deaths']].sum(),axis=1)

new_df = new_df.rename_axis('Date').reset_index()

data_new = pd.melt(new_df,id_vars=['Date'],value_vars=['Confirmed','Deaths','Recovered'],var_name='group_var',value_name='Cases')

data_new = data_new.sort_values(by=['Date','group_var']).reset_index(drop=True)

data_new['label'] = data_new['group_var']

new_data = data_new.pivot_table(index=['Date'], columns='group_var')

new_data.columns = new_data.columns.droplevel().rename(None)

fig,ax = plt.subplots(1,figsize=(16,8))

new_data[['Confirmed','Recovered','Deaths']].plot(ax=ax,fontsize=15)

plt.title(label='Reported Cases In Time',loc='Left',fontsize='20')

ax.set_ylabel('frequency',fontsize=20)

ax.set_ylim([0,30000000])

ax.set_xlabel('')

plt.grid()

plt.tight_layout()
data_new = pd.melt(data[['Date','Country/Region','Confirmed','Recovered','Deaths']],id_vars=['Date','Country/Region'],value_vars=['Confirmed','Recovered','Deaths'],var_name='group_var',value_name='Cases')

data_new['Date'] = pd.to_datetime(data_new['Date'])

new_df = data[['Country/Region','Confirmed','Recovered','Deaths']].groupby(['Country/Region']).sum().reset_index()

data_new = pd.melt(new_df,id_vars=['Country/Region'],value_vars=['Confirmed','Deaths','Recovered'],var_name='group_var',value_name='Cases')

data_new = data_new.sort_values(by=['Country/Region','group_var']).reset_index(drop=True)

new_data = data_new.pivot_table(index=['Country/Region'], columns='group_var')

new_data.columns = new_data.columns.droplevel().rename(None)

fig,ax = plt.subplots(3,figsize=(16,24))

group_labels = ['Confirmed','Deaths','Recovered']

new_data.nlargest(5, ['Confirmed']).plot(y='Confirmed',ax=ax[0],kind='bar')

new_data.nlargest(5, ['Deaths']).plot(y='Deaths',ax=ax[1],kind='bar')

new_data.nlargest(5, ['Recovered']).plot(y='Recovered',ax=ax[2],kind='bar')
data['Province/State'] = data['Province/State'].fillna('Unknown')

data_new = pd.melt(data[['Date','Province/State','Confirmed','Recovered','Deaths']],id_vars=['Date','Province/State'],value_vars=['Confirmed','Recovered','Deaths'],var_name='group_var',value_name='Cases')

data_new['Date'] = pd.to_datetime(data_new['Date'])

new_df = data[['Province/State','Confirmed','Recovered','Deaths']].groupby(['Province/State']).sum().reset_index()

data_new = pd.melt(new_df,id_vars=['Province/State'],value_vars=['Confirmed','Deaths','Recovered'],var_name='group_var',value_name='Cases')

data_new = data_new.sort_values(by=['Province/State','group_var']).reset_index(drop=True)

new_data = data_new.pivot_table(index=['Province/State'], columns='group_var')

new_data.columns = new_data.columns.droplevel().rename(None)

fig,ax = plt.subplots(3,figsize=(16,24))

group_labels = ['Confirmed','Deaths','Recovered']

new_data.nlargest(5, ['Confirmed']).plot(y='Confirmed',ax=ax[0],kind='bar')

new_data.nlargest(5, ['Deaths']).plot(y='Deaths',ax=ax[1],kind='bar')

new_data.nlargest(5, ['Recovered']).plot(y='Recovered',ax=ax[2],kind='bar')
def clean_corpus(text):

    text = re.sub(r'[^\w\s]', '', text)

    text = text.strip()

    text= text.lower()

    stop_words = set(nltk.corpus.stopwords.words('english'))

    word_tokens = nltk.tokenize.word_tokenize(text) 

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

    for w in word_tokens: 

        if w not in stop_words: 

            filtered_sentence.append(w)

    text = ' '.join(filtered_sentence)

    text = "".join(filter(lambda x: not x.isdigit(), text))

    text = text.strip()

    return text
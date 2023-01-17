import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import matplotlib

from sklearn.linear_model import LinearRegression

from gensim import corpora,models

from collections import Counter
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_Analysis=pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')

# preview data

data_Analysis.head()
data_Analysis['Date']=pd.to_datetime(data_Analysis['Date'])

data_Analysis['Year']=data_Analysis['Date'].dt.year

data_Analysis['Day']=data_Analysis['Date'].map(lambda x:x.day)

data_Analysis['Month']=data_Analysis['Date'].map(lambda x:x.month)

data_Analysis.head()
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
plt.figure(figsize=(45,10))

sns.barplot('Year','Aboard',data=data_Analysis)

data_Analysis['Aboard'].head()
# In seaborn, the barplot() function operates on a full dataset and shows an arbitrary estimate, using the mean by default. 

# When there are multiple observations in each category, it also uses bootstrapping to compute a confidence interval around the estimate and plots that using error bars

plt.figure(figsize=(45,10))

sns.barplot('Year','Fatalities',data=data_Analysis)
plt.figure(figsize=(45,10))

sns.stripplot('Year','Fatalities',data=data_Analysis)
plt.figure(figsize=(45,10))

sns.countplot(x='Year', data=data_Analysis, palette="Greens_d");
# Total Aboard and Fatalities plot - for each year

plt.figure(figsize=(100,10))

#Plot1 - background - "total"(top) series

sns.barplot('Year','Aboard',data=data_Analysis,color="blue")

#Plot2 - overlay - "bottom" series

bottom_plot=sns.barplot('Year','Fatalities',data=data_Analysis,color="red")



bottom_plot.set_ylabel("mean(Fatalities) and mean(Aboard)")

bottom_plot.set_xlabel("Year")
crashes_per_year=Counter(data_Analysis['Year'])

years=list(crashes_per_year.keys())

crashes_year=list(crashes_per_year.values())
crashes_per_day=Counter(data_Analysis['Day'])

days=list(crashes_per_day.keys())

crashes_days=list(crashes_per_day.values())
def get_season(month):

    if month >=3 and month <=5:

        return 'spring'

    elif month>=6 and month <=8:

        return 'summer'

    elif month>=9 and month<=11:

        return 'autumn'

    else:

        return 'winter'

    

data_Analysis['Season']=data_Analysis['Month'].apply(get_season)  
crashes_per_season=Counter(data_Analysis['Season'])

seasons=list(crashes_per_season.keys())

crashes_season=list(crashes_per_season.values())
sns.set(style="whitegrid")

sns.set_color_codes("pastel")



fig=plt.figure(figsize=(14,10))



sub1=fig.add_subplot(211)

sns.barplot(x=years,y=crashes_year,color='g',ax=sub1)

sub1.set(ylabel="Crashes",xlabel="Year",title="Plane crashes per year")

plt.setp(sub1.patches,linewidth=0)

plt.setp(sub1.get_xticklabels(),rotation=70,fontsize=9)



sub2=fig.add_subplot(223)

sns.barplot(x=days,y=crashes_days,color='r',ax=sub2)

sub2.set(ylabel="Crashes",xlabel="Day",title="Plane crashes per day")



sub3=fig.add_subplot(224)

sns.barplot(x=seasons,y=crashes_season,color='b',ax=sub3)

texts=sub3.set(ylabel="Crashes",xlabel="Season",title="Plane crashes per season")



plt.tight_layout(w_pad=4,h_pad=3)
survived=[]

dead=[]

for year in years:

    curr_data=data_Analysis[data_Analysis['Year']==year]

    survived.append(curr_data['Aboard'].sum() - curr_data['Fatalities'].sum())

    dead.append(curr_data['Fatalities'].sum())
f,axes=plt.subplots(2,1,figsize=(14,10))

sns.barplot(x=years,y=survived,color='b',ax=axes[0])

axes[0].set(ylabel="Survived",xlabel="Year",title="Survived per year")

plt.setp(axes[0].patches,linewidth=0)

plt.setp(axes[0].get_xticklabels(),rotation=70,fontsize=9)



sns.barplot(x=years,y=dead,color='r',ax=axes[1])

axes[1].set(ylabel="Fatalities",xlabel="Year",title="Dead per year")

plt.setp(axes[1].patches,linewidth=0)

plt.setp(axes[1].get_xticklabels(),rotation=70,fontsize=9)



plt.tight_layout(w_pad=4,h_pad=3)
oper_list=Counter(data_Analysis['Operator']).most_common(12)

operators=[]

crashes=[]

for tpl in oper_list:

    if 'Military' not in tpl[0]:

        operators.append(tpl[0])

        crashes.append(tpl[1])

print('Top 10 the worst operators')

pd.DataFrame({'Count of crashes':crashes},index=operators)

        
loc_list=Counter(data_Analysis['Location'].dropna()).most_common(15)

locs=[]

crashes=[]

for loc in loc_list:

    locs.append(loc[0])

    crashes.append(loc[1])

print('Top 15 the most dangerous locations')

pd.DataFrame({'Crashes in this location':crashes},index=locs)
summary=data_Analysis['Summary'].tolist()

punctuation=['.',',',':']

texts=[]



for text in summary:

    cleaned_text=str(text).lower()

    for mark in punctuation:

        cleaned_text=cleaned_text.replace(mark,'')

    texts.append(cleaned_text.split())
dictionary=corpora.Dictionary(texts)

dictionary.dfs.items()
word_list=[]

for key,value in dictionary.dfs.items():

    if value>100:

        word_list.append(key)
dictionary.filter_tokens(word_list)

corpus=[dictionary.doc2bow(text) for text in texts]
np.random.seed(76543)

lda=models.LdaModel(corpus,num_topics=10,id2word=dictionary,passes=5)
topics=lda.show_topics(num_topics=10,num_words=15,formatted=False)

for topic in topics:

    print('Topic %d:' % topic[0])

    for pair in topic[1]:

        print (pair[0])

    print()
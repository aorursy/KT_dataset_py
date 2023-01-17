

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
## libraries for data wrangling and visualisation are imported

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
Movies = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv')

Movies.head()
Movies.info()
Movies.drop(['Unnamed: 0','ID','Rotten Tomatoes'],axis = 1,inplace = True)
Movies.columns
print('No of features:',Movies.shape[1],'\nNo of Movies:',Movies.shape[0])
Movies.isnull().sum()
Movies[['Directors','Genres','Country','Language','Runtime','Age']] = Movies[['Directors','Genres','Country','Language','Runtime','Age']].fillna('NA')
Movies['Age'].value_counts()
Movies['Age'] = Movies['Age'].str.replace('+','')

Movies['Age'] = Movies['Age'].replace('NA','0')

Movies['Age'] = Movies['Age'].replace('all','1')
Movies['Age'].value_counts()
sns.kdeplot(Movies['IMDb'],shade = True)

plt.axvline(x = Movies['IMDb'].median(),color = 'red',label = 'median')

plt.legend()
Movies['IMDb'].fillna(Movies['IMDb'].median(),inplace = True)
m_count = {'platform':['Netflix','Hulu','Prime Video','Disney+'],

            'MCount':[Movies['Netflix'].sum(),Movies['Hulu'].sum(),Movies['Prime Video'].sum(),Movies['Disney+'].sum()]}



m_count = pd.DataFrame(m_count)
plt.figure(figsize=(8,5))

sns.barplot(x='platform',y='MCount',data = m_count)

plt.xlabel('OTT platform',labelpad = 20)

plt.ylabel('count',labelpad = 20)

plt.show()
Dir = Movies.drop('Directors', axis=1).join(

    Movies['Directors'].str.split(',', expand=True).stack().reset_index(drop=True, level=1).rename('Director'))

D_count = Dir['Director'].value_counts().head(15).reset_index().set_index('index')

D_count = D_count[1:16]

plt.figure(figsize=(8,8))

sns.barplot(x=D_count.index,y=D_count.Director,data = D_count)

plt.xticks(rotation =90)

plt.xlabel('Director')

plt.ylabel('count')

plt.show()
Lang = Movies.drop('Language', axis=1).join(

    Movies['Language'].str.split(',', expand=True).stack().reset_index(drop=True, level=1).rename('Language'))



Lang_count = Lang['Language'].value_counts().head(25).reset_index().set_index('index')
plt.figure(figsize=(8,4))

sns.barplot(x=Lang_count.index,y=Lang_count.Language,data = m_count)

plt.xticks(rotation =90)

plt.xlabel('Language')

plt.ylabel('count')

plt.show()
Genre = Movies.drop('Genres', axis=1).join(

    Movies['Genres'].str.split(',', expand=True).stack().reset_index(drop=True, level=1).rename('Genre'))

Genre_count = Genre['Genre'].value_counts().reset_index().set_index('index')
plt.figure(figsize=(8,8))

sns.barplot(x=Genre_count.index,y=Genre_count.Genre,data = m_count)

plt.xticks(rotation =90)

plt.xlabel('Genre')

plt.ylabel('count')

plt.show()
lis = []

for i in range(0,Genre.shape[0]):

    lis.append(Genre.iloc[i,13])

    

from collections import Counter

G_count = Counter(lis)



from wordcloud import WordCloud

wc = WordCloud(background_color='white')

wc.generate_from_frequencies(G_count)

plt.figure(figsize=(12,10))

plt.imshow(wc,interpolation='bilinear')

plt.axis('off')

plt.show()
L_Netflix = Lang.loc[Lang['Netflix'] == 1,'Language'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

L_Prime =  Lang.loc[Lang['Prime Video'] == 1,'Language'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

L_Hulu = Lang.loc[Lang['Hulu'] == 1,'Language'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

L_Disney = Lang.loc[Lang['Disney+'] == 1,'Language'].value_counts().reset_index().set_index('index').drop('NA',axis =0)
fig , axes = plt.subplots(2,2,figsize = (12,12))

 

plt.subplots_adjust(hspace = 0.6,wspace = 0.5)    

    

L_Netflix.head(10).plot(kind = 'bar',ax = axes[0,0])

axes[0,0].set_title('Netflix')

axes[0,0].set_xlabel('')

axes[0,0].set_ylabel('')



L_Prime.head(10).plot(kind = 'bar',ax = axes[0,1])

axes[0,1].set_title('Prime Video')

axes[0,1].set_xlabel('')

axes[0,1].set_ylabel('')





L_Hulu.head(10).plot(kind = 'bar',ax = axes[1,0])

axes[1,0].set_title('Hulu')

axes[1,0].set_xlabel('')

axes[1,0].set_ylabel('')







L_Disney.head(10).plot(kind = 'bar',ax = axes[1,1])

axes[1,1].set_title('Disney')



axes[1,1].set_xlabel('')

axes[1,1].set_ylabel('')



fig.text(0.5, 0.004, 'Language', ha='center',fontsize = 'large')

fig.text(0.004, 0.5, 'Count', va='center', rotation='vertical',fontsize = 'large')

plt.show()
L_ratings = Lang.groupby('Language')['IMDb'].median()

L_ratings = L_ratings.reset_index().set_index('Language')
Top_10_lang = L_ratings.loc[['English','Hindi','Spanish','French','German','Italian'

                                                      ,'Japanese','Korean','Mandarin','Russian'],'IMDb']

Top_10_lang
Top_10_lang = Top_10_lang.reset_index().set_index('Language')

English = Lang.loc[Lang['Language']=='English','IMDb'].reset_index().set_index('index')
fig,axes = plt.subplots(1,2,figsize = (18,6))



sns.kdeplot(English['IMDb'],ax = axes[1],shade = True)

plt.axvline(English['IMDb'].median(),color = 'red')



sns.barplot(x=Top_10_lang.index,y=Top_10_lang['IMDb'],ax = axes[0])

plt.show()
G_Netflix = Genre.loc[Genre['Netflix'] == 1,'Genre'].value_counts().reset_index().set_index('index')

G_Prime =  Genre.loc[Genre['Prime Video'] == 1,'Genre'].value_counts().reset_index().set_index('index')

G_Hulu = Genre.loc[Genre['Hulu'] == 1,'Genre'].value_counts().reset_index().set_index('index')

G_Disney = Genre.loc[Genre['Disney+'] == 1,'Genre'].value_counts().reset_index().set_index('index')
fig , axes = plt.subplots(2,2,figsize = (12,12))



 

G_Netflix.head(10).plot(kind = 'bar',ax = axes[0,0],color = 'brown')

axes[0,0].set_title('Netflix')

axes[0,0].set_xlabel('')

axes[0,0].set_ylabel('')



G_Prime.head(10).plot(kind = 'bar',ax = axes[0,1],color = 'green')

axes[0,1].set_title('Prime Video')

axes[0,1].set_xlabel('')



G_Hulu.head(10).plot(kind = 'bar',ax = axes[1,0],color = 'gray')

axes[1,0].set_title('Hulu')

axes[1,0].set_xlabel('')

axes[1,0].set_ylabel('')





G_Disney.head(10).plot(kind = 'bar',ax = axes[1,1])

axes[1,1].set_title('Disney')

axes[1,1].set_xlabel('')



plt.tight_layout()

fig.text(0.5, 0.004, 'Genre', ha='center',fontsize = 'large')

fig.text(0.004, 0.5, 'Count', va='center', rotation='vertical',fontsize = 'large')

plt.show()
G_ratings = Genre.groupby('Genre')['IMDb'].median()

G_ratings = G_ratings.reset_index().set_index('Genre')
Top_10_genre = G_ratings.loc[['Drama','Comedy','Thriller','Action','Romance','Crime','Adventure','Horror','Family','Mystery'],'IMDb']

Top_10_genre = Top_10_genre.reset_index().set_index('Genre')
plt.figure(figsize=(8,5))

sns.barplot(x=Top_10_genre.index,y=Top_10_genre['IMDb'])

plt.xticks(rotation = 90)

plt.show()
D_Netflix = Dir.loc[Dir['Netflix'] == 1,'Director'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

D_Prime =  Dir.loc[Dir['Prime Video'] == 1,'Director'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

D_Hulu = Dir.loc[Dir['Hulu'] == 1,'Director'].value_counts().reset_index().set_index('index').drop('NA',axis =0)

D_Disney = Dir.loc[Dir['Disney+'] == 1,'Director'].value_counts().reset_index().set_index('index').drop('NA',axis =0)
fig,axes = plt.subplots(2,2,figsize=(12,12))

D_Netflix.head(10).plot(kind = 'bar',ax = axes[0,0],color = 'brown')

axes[0,0].set_title('Netflix')

axes[0,0].set_xlabel('')

axes[0,0].set_ylabel('')





D_Prime.head(10).plot(kind = 'bar',ax = axes[0,1],color = 'green')

axes[0,1].set_title('Prime Video')

axes[0,1].set_xlabel('')

axes[0,1].set_ylabel('')





D_Hulu.head(10).plot(kind = 'bar',ax = axes[1,0],color = 'gray')

axes[1,0].set_title('Hulu')

axes[1,0].set_xlabel('')

axes[1,0].set_ylabel('')







D_Disney.head(10).plot(kind = 'bar',ax = axes[1,1])

axes[1,1].set_title('Disney')

axes[1,1].set_xlabel('')

axes[1,1].set_ylabel('')



fig.tight_layout()

fig.text(0.5, 0.004, 'Director', ha='center',fontsize = 'large')

fig.text(0.004, 0.5, 'Count', va='center', rotation='vertical',fontsize = 'large')

plt.show()
D_ratings = Dir.groupby('Director')['IMDb'].median()

D_ratings = D_ratings.reset_index().set_index('Director')
Top_10_dir = D_ratings.loc[['Jay Chapman','Joseph Kane','Cheh Chang','Jim Wynorski','William Beaudine','Sam Newfield','David DeCoteau','Jay Karas','Marcus Raboy','William Witney'],'IMDb']

Top_10_dir = Top_10_dir.reset_index().set_index('Director')
plt.figure(figsize=(8,5))

sns.barplot(x=Top_10_dir.index,y=Top_10_dir['IMDb'])

plt.xticks(rotation = 90)

plt.xlabel('Director',labelpad= 20)

plt.ylabel('IMDb',labelpad = 20)

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,7))

sns.distplot(Movies['IMDb'],bins=20,kde=True,color='r',ax=ax[0])

sns.boxplot(Movies['IMDb'],ax=ax[1],color='r',saturation=0.5)

plt.show()
country = Movies[Movies['Country']=='India']

country = country.drop('Language', axis=1).join(

    country['Language'].str.split(',', expand=True).stack().reset_index(drop=True, level=1).rename('Language'))
Top_10_lang_india = country['Language'].value_counts().head(10).reset_index().set_index('index')
plt.figure(figsize=(8,5))

sns.barplot(x=Top_10_lang_india.index,y=Top_10_lang_india['Language'])

plt.xticks(rotation = 90)

plt.xlabel('Language',labelpad= 20)

plt.ylabel('count',labelpad = 20)

plt.show()
Im_count = {'platform':['Netflix','Hulu','Prime Video','Disney+'],

            'ImCount':[country['Netflix'].sum(),country['Hulu'].sum(),country['Prime Video'].sum(),country['Disney+'].sum()]}



Im_count = pd.DataFrame(Im_count)
plt.figure(figsize=(8,4))

sns.barplot(x='platform',y='ImCount',data = Im_count)

plt.xlabel('OTT platform')

plt.ylabel('count')

plt.show()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
folders=['/kaggle/working/img','/kaggle/working/img/turkish','/kaggle/working/img/arabic' ]
for folder in folders:
    if not os.path.exists(folder):
        os.mkdir(folder)
df=pd.read_excel("/kaggle/input/quran-words-with-meaning-wikipedia-info/QURAN.xlsx",index_col=0)
df.head()
df.info()
revelation=[96,68,73,74,1,111,81,87,92,89,93,94,103,100,108,102,107,109,105,113,114,112,53,80,97,91,85,95,106,101,75,104,77,50,90,86,54,38,7,72,36,52,35,19,20,56,26,27,28,17,10,11,12,15,6,37,31,34,39,40,41,42,43,44,45,46,51,88,18,16,71,14,21,23,32,52,67,69,70,78,79,82,84,30,29,83,2,8,3,33,60,4,99,57,47,13,55,76,65,98,59,24,22,63,58,49,66,64,61,62,48,5,9,110]
#set nüzul order
df['Revelation'] = df['SURAH'].apply(lambda x: revelation[x-1])
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 25})
sns.set_context("paper")

plt.scatter(df.SURAH, df.Revelation, s=50,
            alpha=0.2, cmap="viridis") 
plt.colorbar(); 
plt.xlabel("SURAH") 
plt.ylabel("Revelation") 
plt.title("Relationship between SURAH and Revelation") 
plt.show()
revs=df.Revelation.unique()
plt.hist([df.loc[df.Revelation == x, 'VERSE'] for x in revs], label=revs)
plt.show()
fig, ax = plt.subplots()
ax.hist(df.Revelation, label="Revelation", bins=20)
ax.set_xlabel("Revelation")
ax.set_ylabel("Number of SURAH")
plt.show()
dfChapter=pd.read_excel("/kaggle/input/quran-words-with-meaning-wikipedia-info/WikiChapters.xlsx", index_col=None)
dfChapter.head()
import re
dfChapter['Revelation'] = pd.to_numeric(dfChapter['ROW_NUM'].apply(lambda x: revelation[x-1]))
dfChapter['VERSE_COUNT'] = pd.to_numeric(dfChapter['VERSES_DESC'].apply(lambda x: re.search(r'\d*\a*\d*', x)[0]))
dfChapter.head()
sns.relplot(x="ROW_NUM", y="VERSE_COUNT",data=dfChapter, hue="PLACE_OF_REV", kind="scatter")
plt.show()
g=sns.relplot(x="ROW_NUM", y="VERSE_COUNT", hue="PLACE_OF_REV",aspect=4, palette=["b","r"],data=dfChapter, kind="line",  ci="sd",row="PLACE_OF_REV")
plt.xlabel="SURAH_NO"
plt.xticks(rotation=90)
plt.show()
dfMerge=df.merge(dfChapter,left_on=['Revelation','SURAH'],right_on=['Revelation','ROW_NUM'])
dfMerge['Counts'] = np.zeros(len(df))
dfMeaning= dfMerge.groupby(['Revelation','TITLE','MEANING_GROUP','ROW_NUM','BRX_ROOT','PLACE_OF_REV'], as_index=False)['Counts'].count()
dfMeaning.head()
import matplotlib.pyplot as plt
places=["Makkah","Madinah"]
criterias=["Kadın","Erkek"]
colors=["#648FFF","#DC267F","#FFB000","#00B000"]
fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
plt.xticks(rotation=60)
i=0
for row in axs:
    j=0
    for ax in row:    
        dfSub=dfMeaning[(dfMeaning['MEANING_GROUP']==criterias[i] )& (dfMeaning.PLACE_OF_REV==places[j])]
        ax.plot(dfSub.TITLE,dfSub.Counts, colors[i*2+j])
        ax.set_xlabel(criterias[i]+"/"+places[j])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        j+=1
    i+=1
plt.show()
from PIL import Image
from wordcloud import WordCloud
exclude_words = ['Şimdi','gelmek','demek','إِنْ','إِنَّ','لَـمْ','لَــمَّا','إِنَّ','لِماَ','Bu','etmek','Etmek','Kim',                      
'Gelmek','demek','bilmek','Bilmek','dilemek','Dilemek','Şu','Bunlar','Şunlar','Gibi','Öyle ki','içinde',
'Önce','Onlar','-e, -a','dan','-dan',"-den",'den','-den -dan','Bazı','Ya da','değil',"Değil",'Söylemek',
'Öyle ki','Dek','Veya','ve','Ya','Ya Da',"olmak", "etmek",'veya','İçin','Eğer','önce','Ey','Sen','Ben','Biz','siz',
'biz','ben','O','için','öyle','gel','-de','sen','ey','bu','şu','o','bunlar','şunlar','onlar','Hani','İçinde','kez',
'az','Az','çok','Çok','en','ki','içinde','Eğer','üzerine','Öyle','yapmak','Fakat','Ama','Lakin','ancak','Ancak',
'ile','İle','-ki','bir','Başka','önce','sonra','arasında','Çok','Az','Sonra','söylemek','görmek','belki']
exclude_words_arabic=[]
for word in exclude_words:
    arabic_words=str(dfMerge.loc[dfMerge['MEANING_GROUP']==word]['BRX_ROOT'].unique()).replace('[','').replace(']','')
    #print(f'{word} T:{type(arabic_words)} arabic: {arabic_words} l:{len(arabic_words)} ls: {len(arabic_words.split())}')
    if(arabic_words!='[]'):
        for arabic_word in arabic_words.split():
            exclude_words_arabic.append(arabic_word.strip("'"))
            
exclude_words_arabic=np.unique(exclude_words_arabic)            
  

verses = {}
for i in range(1,115):
    verses[str(i)] = " ".join('' if mean in (exclude_words) else mean for mean in dfMerge[dfMerge['SURAH']==i]['MEANING_GROUP'])

for key in verses.keys():
    print(dfChapter.loc[int(key)-1]['TITLE'])
    wordcloud = WordCloud(background_color='black', width=800, height=800).generate(verses[key])
    plt.figure( figsize=(20,10) )
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
    """filename="/kaggle/working/img/turkish/"+key+'-'+dfChapter.loc[int(key)-1]['TITLE']+'.png'
    wordcloud.to_file(filename)"""
   
!pip install arabic_reshaper
import arabic_reshaper
!pip install python-bidi
from bidi.algorithm import get_display
verses_arabic = {}
from wordcloud import WordCloud
font_file='/kaggle/input/arabic-character-format/NotoNaskhArabic-Regular.ttf'


for i in range(1,115):
    verses_arabic[str(i)] = " ".join('' if root in (exclude_words_arabic) else root for root in dfMerge[dfMerge['SURAH']==i]['BRX_ROOT'])
    verses_arabic[str(i)]= arabic_reshaper.reshape(verses_arabic[str(i)] )
    verses_arabic[str(i)]= get_display(verses_arabic[str(i)])

for key in verses_arabic.keys():
    print(dfChapter.loc[int(key)-1]['TITLE'])
    wordcloud = WordCloud(font_path=font_file, background_color='black', width=800, height=800).generate(verses_arabic[key])
    plt.figure( figsize=(20,10) )
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    filename="img/arabic/"+key+'-'+dfChapter.loc[int(key)-1]['TITLE']+'.png'
    wordcloud.to_file(filename)
dfThemes=pd.read_excel("/kaggle/input/themes/Themes.xlsx", index_col=[3,5]) #SURAH ndx and VERSES info index..
dfThemes.head()

dfSurahVerses  ={}
excludedlist=[]
def parse_str_interval(str) :
    parse_list=[]
    for part in str.split(','):
        if(part.find('-')>=0) :
            start,finish=part.split('-')
            for i in range(int(start),int(finish)+1) :
                parse_list.append(i)
        else:
            parse_list.append(int(part))
    return parse_list
                    
for index, row in dfThemes.iterrows():
    verses=index[1].replace(' ','').replace(')','')
    if(verses=='Tamamı') :
        verses_count=(dfChapter.loc[index[0]-1]['VERSE_COUNT'])
        for i in range(1,verses_count+1) :
            dfSurahVerses[index[0],i]=row
    else:
        verse_list=parse_str_interval(verses)
        excluded_list=parse_str_interval(row['EXCLUDES'])        
        for verse_part in verse_list:
            if(verse_part not in excludedlist) :
                dfSurahVerses[index[0],verse_part]=row

dfSurahVerses=pd.DataFrame.from_dict(dfSurahVerses, orient='index')
print(dfSurahVerses.info())
import seaborn as sns
sns.catplot(data=dfSurahVerses, x='YEAR', y='GRUP', hue='PLACE')
sns.set_context("paper")
plt.scatter(dfSurahVerses['YEAR'], dfSurahVerses['GRUP'], c= [1 if x =='Medine' else 0 for x in dfSurahVerses['PLACE']] , alpha=0.2, cmap="viridis")
plt.colorbar(); 
ax.set_xlabel("YEAR")
ax.set_ylabel("Theme")
ax.set_title("Acc. Year Themes")
plt.show()
!pip install pandas plotnine

from plotnine import *
"""facet_grid('PLACE',scales='free_x')+ \ """
ggplot(dfSurahVerses, aes(x='YEAR', color='GRUP')) + \
    geom_line(stat = 'count') + \
    facet_wrap(['PLACE'],scales='free_y') 
#    flip_xlabels + \


import seaborn as sns
g = sns.FacetGrid(dfSurahVerses, col="PLACE", col_wrap=2, height=8, 
                  hue_order=["Mekke", "Medine"],
                  hue_kws=dict(marker=["^", "v"]) )
g = g.map(plt.plot, "YEAR","GRUP")

ax = sns.swarmplot(y="GRUP", x="YEAR", hue="PLACE", data=dfSurahVerses, palette="Set2", dodge=True)
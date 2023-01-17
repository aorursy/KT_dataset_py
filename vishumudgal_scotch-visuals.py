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
import re
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import HTML
import plotly as py
import warnings

warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)
HTML('''
<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
df=pd.read_csv('/kaggle/input/22000-scotch-whisky-reviews/scotch_review.csv')
df
df.shape
df.isnull().sum()
col=df.columns
nunique=df.nunique()
nul = df.isnull().sum()
typ = df.dtypes
nuniq = pd.DataFrame({'column_name':col,'no. of Unique Values':nunique,'Missing Values':nul,'Class Type':df.dtypes})
nuniq.reset_index(drop=True)
print(df['Unnamed: 0'])
# These all are the unique numbers 
#so let us rename this column
df=df.rename(columns={'Unnamed: 0':'ID'})
df
df['currency']
df = df.drop(['currency'],axis=1)
df.head()
print(df['category'].head())
print('\n\n Type of class of this feature is :',df.category.dtype)
print('\n\n number of unique category : ',df.category.nunique())
df.category.unique()
plt.figure(figsize=(10,7))
a=sns.countplot(df['category'],palette='rocket_r')
_=plt.xticks(rotation=70)
plt.show()
print('Class type of the column \t',df['review.point'].dtype)
print('number of the unique values : ',df['review.point'].nunique())
print("\n Let's check out the range of points given\t",df['review.point'].min(),"-",df['review.point'].max())

def fig(length,width):
    plt.figure(figsize=(length,width))

def setit(x):
    _=plt.xticks(rotation=x)
    plt.show()
fig(10,5)
sns.catplot('category','review.point',data=df,palette='Oranges_r')
setit(70)
fig(12,5)
sns.swarmplot(x=df['category'],y=df['review.point'])
setit(70)
fig(12,7)
sns.boxplot(x=df['category'],y=df['review.point'],data=df,palette='Purples_r',saturation=.6,fliersize=8,whis=2)
setit(70)
sns.lineplot('category','review.point',data=df)
setit(30)
fig(7,5)
sns.scatterplot('category','review.point',data=df)
plt.ylim(90,98)
setit(30)
#take the isight of price cloumn
#df['price'].unique()
int_price=[]

# finding out the complications in the column
for i in df['price']:
    #removing $ and , so that we can convert this feature price into integer type
    _=re.sub(r'[$,]+','',i)
    #converting float integrs
    x=re.sub(r'\W\d\d','',_)
    z=0
    #converting liter into one botle price
    if ("/l" in x):
        l=re.sub(r'[/l]\w+','',x)
        
        z=int(l)
        z=z*.75
        int_price.append(z)
        
    # if any of the alphanumeric value like space like we encountered the case : ('$15,000 or $60,000/set')     
    elif(" " in  x):
        l= re.sub(r'[ ]\w+\W+\w+','',x)
        z=int(l)
        int_price.append(z)
            
    elif ("set" in x):
        l=re.sub(r'[/]\w+','',x)
        
        z= int(l)
        z=z/4
        int_price.append(z)
    else :
        z=int(x)
        int_price.append(z)
#print(int_price)             
df['price'] = int_price
df['price'].dtype
fig(10,5)
sns.set(color_codes=True)
sns.lineplot(df['category'],df['price'],data=df)
setit(10)
sns.set(color_codes=True)
sns.lineplot(y=df.price, x=df.index , data=df)
plt.xlabel('Frequency');
sns.lineplot(df['review.point'],df['price'],data=df)
plt.xlim(60,100)
plt.ylim(0,15000)
fig(12,7)
sns.lmplot('review.point','price',data=df,hue='category')
_=plt.xlim(65,100)
_=plt.ylim(0,18000)
df.sort_values(by = 'review.point' , ascending = False)[['name','category','review.point','price' ]].head(15)
avp=df['price'].mean()
avr=df['review.point'].mean()
print('Average price of all the scotch in data set is: \t',avp)
print('Average review point of all the scotch in data set is: \t',avr)
# scotch under $50 with review point above 90
_50=df[(df['review.point'] > 90) & (df['price'] < 50)]
_50
#scotch under $100 with rating above 95
_100=df[(df['review.point'] > 95) & (df['price'] < 100)]
_100
#scotch above $100 but review point below 76
#worst scotch in the list
_100_=df[(df['review.point'] < 76) & (df['price'] > 100)]
_100_
# avp/avr ratio
# df['ratio'] will show us the
ratio = df['price']/df['review.point']
z=pd.DataFrame({'name':df['name'],'marks':df['review.point'],'price':df['price'],'category':df['category'],'ratio':ratio})

#price efficient scotch
z[z['ratio']<.2]
sns.distplot(z['ratio'])
plt.ylim(0,0.00115)
df['name'].head(5)
#clearing the unwanted string present in ()
clean = []
for i in df.name:
    if ("(" in i):
        x  =re.sub( r'\([^)]*\)','',i )
        clean.append(x)
    else : 
        clean.append(i)

    
      
        
df['name'] = clean
#remove head() and take the insight of name
df['name'].head()
review=df.copy()
# finding the alchohol content percentage of each of the bottle of scotch
alc=[]
for i in df['name']:
    per = re.findall(r'(\d\d\W?\d?%)',i)
    if (len(per) == 0):
        _ = float('NaN')
        alc.append(_)
    else:
        if(len(per)==2):
            __=re.sub('%','',per[1])
            _=float(__)
            alc.append(_)
        else:
            __ = re.sub('%','',per[0])
            _ = float(__)
            alc.append(_)
df['percentage']=alc
clean =[]
# removing the alchohol percentage from name
for i in df['name']:
    _ = re.sub(r'(\d\d\W?\d?%)','',i)
    clean.append(_)
df['name']=clean
# finding the age of the bottle
age=[]
for i in df['name']:
    yod = re.findall('\d\d? year old',i)
    if (len(yod) == 1):
        __ = re.findall('\d\d?',yod[0])
        ag=int(__[0])
        age.append(ag)
    else:
        __ = re.findall(' \d\d\d\d ',i)
        if (len(__)== 1):
            _=int(__[0])
            ag=2020-_
            if (ag>200):
                ag=float("NaN")
                age.append(ag)
            else :
                age.append(ag)
        else:
            ag=float('NaN')
            age.append(ag)
    
df['age'] = age

clean=[]
#removing the age
for i in df['name']:
    _ = re.sub(' \d\d? year old','',i)
    clean.append(_)
df['name'] = clean    
#final cleaning of df['name']
clean=[]
for i in df['name']:
    i=re.sub(r'[,]+','',i)
    i=re.sub(r'  ',' ',i)
    clean.append(i)
df['name'] = clean
df
df.sort_values(by = 'percentage' , ascending = False)[['name','category','review.point','price','age','percentage' ]].head(5)
df.sort_values(by = 'price' , ascending = False)[['name','category','review.point','price','age','percentage' ]].tail(5)
df.sort_values(by = 'price' , ascending = False)[['name','category','review.point','price','age','percentage' ]].head(5)
fig(12,7)
sns.pointplot(x='marks',y='ratio',data=z,join=False,dodge=True,palette='inferno',hue='category',markers='x')

fig, ax = plt.subplots(figsize=(16,10), facecolor='white', dpi= 80)
ax.vlines(x=z.index, ymin=0, ymax=z.ratio, color='firebrick', alpha=0.7, linewidth=20)

p1 = patches.Rectangle((.57, -0.005), width=.33, height=.13, alpha=.1, facecolor='green', transform=fig.transFigure)
p2 = patches.Rectangle((.124, -0.005), width=.446, height=.13, alpha=.1, facecolor='red', transform=fig.transFigure)
fig.add_artist(p1)
fig.add_artist(p2)
plt.show()
#grouping
x_var = 'ratio'
groupby_var = 'category'
z_agg = z.loc[:, [x_var, groupby_var]].groupby(groupby_var)
vals = [z[x_var].values.tolist() for i, z in z_agg]

#draw
plt.figure(figsize=(16,9), dpi= 80)
colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
n, bins, patches = plt.hist(vals, 30, stacked=True, density=False, color=colors[:len(vals)])

# Decoration
plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
plt.xlabel(x_var)
plt.ylabel("Frequency")
plt.ylim(0, 20)
plt.xticks(ticks=bins[::3], labels=[round(b,1) for b in bins[::3]])
plt.show()



corr = df.corr()
plt.subplots(figsize=(15,12))
sns.heatmap(corr, vmax=0.9, cmap="Greens", square=True)
#most comon brands
figure = plt.figure(figsize=(14,12))
sns.barplot(y=df['name'].value_counts().index[:20], x=df['name'].value_counts()[:20])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xs=df['review.point'].where(df['review.point']>90)
ys=df['age'].where(df['age']>45)
zs=df['price'].where(df['price'] < 1000)
                            
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')

ax.set_xlabel('review points')
ax.set_ylabel('age')
ax.set_zlabel('Price')


                            
SMS=review[review.category=="Single Grain Whisky"]['description'].values
BSW=review[review.category=='Blended Scotch Whisky']['description'].values
BMSW=review[review.category=='Blended Malt Scotch Whisky']['description'].values
SGW=review[review.category=='Single Grain Whisky']['description'].values
GSW=review[review.category=='Grain Scotch Whisky']['description'].values
#Code inspiration:https://www.kaggle.com/duttadebadri/analysing-the-olympics-for-last-120-yrs/notebook & Nick Brooks from comments ..

from wordcloud import WordCloud,STOPWORDS
stopwords=set(STOPWORDS)
def show_wordcloud(data,title=None):
    wc=WordCloud(background_color="black", max_words=10000,stopwords=STOPWORDS, max_font_size= 40)
    wc.generate(" ".join(data))
    fig=fig = plt.figure(figsize=[8,5], dpi=80)
    plt.axis('off')
    if title:
        fig.suptitle(title,fontsize=16)
        fig.subplots_adjust(top=1)
        plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=1,interpolation='bilinear')
        plt.show()
        
show_wordcloud(SMS,title="Wordcloud for Single Grain Whisky")
show_wordcloud(BSW,title="Wordcloud for Blended Scotch Whisky")
show_wordcloud(BMSW,title="Wordcloud for Blended Malt Scotch Whisky")
show_wordcloud(SGW,title="Wordcloud for Single Grain Whisky")

show_wordcloud(GSW,title="Wordcloud for Grain Scotch Whisky")
df['description'].describe
import nltk
import string

!pip install nlppreprocess
from nlppreprocess import NLP

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
#Converting reviews into lowercase
def lowercase_text(text):
    text = text.lower()
    return text
df['description'] = df['description'].apply(lambda x :lowercase_text(x))
# removing all the unwanted noise (if any)
def remove_noise(text):
    # Dealing with Punctuation
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
df['description'] = df['description'].apply(lambda x :remove_noise(x))
# removing the stop words
nlp = NLP()
df['description'] = df['description'].apply(nlp.process)
#stemming 

stemmer = SnowballStemmer("english")
 
def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)
df['description']=df['description'].apply(stemming)
# vectorization
count_vectorizer = CountVectorizer(analyzer='word', binary=True)
count_vectorizer.fit(df['description'])
cv = count_vectorizer.fit_transform(df['description'])





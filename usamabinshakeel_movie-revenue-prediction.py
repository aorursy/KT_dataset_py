#Importing necessary dependancies
import pandas as pd
import numpy as np
import ast
from pandas.io.json import json_normalize
import json
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from xgboost import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from numpy import mean
from numpy import std
from numpy import asarray
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# This block of code is unnecessary(needed for google colab)
#from google.colab import drive
#drive.mount('/content/drive')
train=pd.read_csv("../input/tmdb-box-office-prediction/train.csv")
test=pd.read_csv("../input/tmdb-box-office-prediction/test.csv")
train.shape,test.shape
train.columns,test.columns
train.info(),test.info()
train.describe()
#Number of missing values in each column  
train.isna().sum()
#code block copied from kaggle. Similar code has also been used further. It is to extract information from columns having JSON type values
train.genres=train.genres.fillna('[{}]')

genresList=[]
for index,row in train.genres.iteritems():
    genresStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            genresStr=genresStr+','+dic['name'] 
    genresStr=genresStr.strip(',') # trim leading ,
    genresList.append(genresStr)
    
tempDF=pd.DataFrame(genresList,columns=['genres'])
train.genres=tempDF['genres']
test.genres=test.genres.fillna('[{}]')

genresList=[]
for index,row in test.genres.iteritems():
    genresStr=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            genresStr=genresStr+','+dic['name'] 
    genresStr=genresStr.strip(',') # trim leading ,
    genresList.append(genresStr)
    
tempDF=pd.DataFrame(genresList,columns=['genres'])
test.genres=tempDF['genres']

train['num_of_genres']=train.genres.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#train = train.join(train.genres.str.get_dummies(','))
test['num_of_genres']=test.genres.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#test= test.join(test.genres.str.get_dummies(','))
#Finding most common values in the given column and manually doing one hot encoding.Similar method has been employed for further columns
Gen=train['genres'].str.cat(sep=',')
words = Gen.split(",")  
Gen1  = Counter(words).most_common(15)
Gen1
Dum=[]
for i in range(0,15):
  e=Gen1[i][0]
  Dum.append(e)
for gen in Dum:
    train['genre_' + gen] = train['genres'].apply(lambda x: 1 if gen in x else 0)
    test['genre_' + gen] = test['genres'].apply(lambda x: 1 if gen in x else 0)

train.production_companies=train.production_companies.fillna('[{}]')

List=[]
for index,row in train.production_companies.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['production_companies'])
train.production_companies=tempDF['production_companies']
test.production_companies=test.production_companies.fillna('[{}]')

List=[]
for index,row in test.production_companies.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['production_companies'])
test.production_companies=tempDF['production_companies']


train['num_of_productioncompanies']=train.production_companies.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#train = train.join(train.production_companies.str.get_dummies(','))

test['num_of_productioncompanies']=test.production_companies.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#test= test.join(test.production_companies.str.get_dummies(','))
Com=train['production_companies'].str.cat(sep=',')
words = Com.split(",")  
Com1  = Counter(words).most_common(15)
Dum=[]
for i in range(0,15):
  e=Com1[i][0]
  Dum.append(e)
for com in Dum:
    train['Company_' + com] = train['production_companies'].apply(lambda x: 1 if com in x else 0)
    test['Company_' + com] = test['production_companies'].apply(lambda x: 1 if com in x else 0)
train.production_countries=train.production_countries.fillna('[{}]')

List=[]
for index,row in train.production_countries.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['production_countries'])
train.production_countries=tempDF['production_countries']
test.production_countries=test.production_countries.fillna('[{}]')

List=[]
for index,row in test.production_countries.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['production_countries'])
test.production_countries=tempDF['production_countries']
train['num_of_productioncountries']=train.production_countries.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
test['num_of_productioncountries']=test.production_countries.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#train = train.join(train.production_countries.str.get_dummies(','))
#test = test.join(test.production_countries.str.get_dummies(','))


Cou=train['production_countries'].str.cat(sep=',')
words = Cou.split(",")  
Cou1  = Counter(words).most_common(15)
Dum=[]
for i in range(0,15):
  e=Cou1[i][0]
  Dum.append(e)
for cou in Dum:
    train['Country_' + cou] = train['production_countries'].apply(lambda x: 1 if cou in x else 0)
    test['Country_' + cou] = test['production_countries'].apply(lambda x: 1 if cou in x else 0)
train.spoken_languages=train.spoken_languages.fillna('[{}]')

List=[]
for index,row in train.spoken_languages.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['spoken_languages'])
train.spoken_languages=tempDF['spoken_languages']
test.spoken_languages=test.spoken_languages.fillna('[{}]')

List=[]
for index,row in test.spoken_languages.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['spoken_languages'])
test.spoken_languages=tempDF['spoken_languages']


train['num_of_spokenlanguages']=train.spoken_languages.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#train = train.join(train.spoken_languages.str.get_dummies(','))

test['num_of_spokenlanguages']=test.spoken_languages.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
#test= test.join(test.spoken_languages.str.get_dummies(','))
Lan=train['spoken_languages'].str.cat(sep=',')
words = Lan.split(",")  
Lan1  = Counter(words).most_common(15)
Dum=[]
for i in range(0,15):
  e=Lan1[i][0]
  Dum.append(e)
for lan in Dum:
    train['Language_' + lan] = train['spoken_languages'].apply(lambda x: 1 if lan in x else 0)
    test['Language_' + lan] = test['spoken_languages'].apply(lambda x: 1 if lan in x else 0)
train.Keywords=train.Keywords.fillna('[{}]')

List=[]
for index,row in train.Keywords.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['Keywords'])
train.Keywords=tempDF['Keywords']
test.Keywords=test.Keywords.fillna('[{}]')

List=[]
for index,row in test.Keywords.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['Keywords'])
test.Keywords=tempDF['Keywords']

train['num_of_keywords']=train.Keywords.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
test['num_of_keywords']=test.Keywords.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
key=train['Keywords'].str.cat(sep=',')
words = key.split(",")  
key1  = Counter(words).most_common(15)
Dum=[]
for i in range(0,15):
  e=key1[i][0]
  Dum.append(e)
for Key in Dum:
    train['Keyword_' + Key] = train['Keywords'].apply(lambda x: 1 if Key in x else 0)
    test['Keyword_' + Key] = test['Keywords'].apply(lambda x: 1 if Key in x else 0)
#Avoid this code

#train.cast=train.cast.fillna('[{}]')

#List=[]
#for index,row in train.cast.iteritems():
    #Str=''
    #listofDict=ast.literal_eval(row)
    #for dic in listofDict:
        
        #if('name' in dic.keys()):
            #string=str(dic['gender'])
            #Str=Str+','+string
    #Str=Str.strip(',') # trim leading ,
    #List.append(Str)
    #print(List)
    
#tempDF=pd.DataFrame(List,columns=['Gender'])
#train['Gender']=tempDF['Gender']
#train = train.join(train.Gender.str.get_dummies(','))
train.cast=train.cast.fillna('[{}]')

List=[]
for index,row in train.cast.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['cast'])
train.cast=tempDF['cast']
test.cast=test.cast.fillna('[{}]')

List=[]
for index,row in test.cast.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['cast'])
test.cast=tempDF['cast']
train['total_castmembers']=train.cast.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
test['total_castmembers']=test.cast.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
train.crew=train.crew.fillna('[{}]')

List=[]
for index,row in train.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_members'])
train['crew_members']=tempDF['crew_members']
test.crew=test.crew.fillna('[{}]')

List=[]
for index,row in test.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('name' in dic.keys()):
            Str=Str+','+dic['name'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_members'])
test['crew_members']=tempDF['crew_members']

train['total_crew_members']=train.crew_members.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)
test['total_crew_members']=test.crew_members.apply(lambda x : len(list(x.split(','))) if x != '[{}]' else 0)

List=[]
for index,row in train.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('department' in dic.keys()):
            Str=Str+','+dic['department'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_department'])
train['crew_department']=tempDF['crew_department']
List=[]
for index,row in test.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('department' in dic.keys()):
            Str=Str+','+dic['department'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_department'])
test['crew_department']=tempDF['crew_department']
List=[]
for index,row in train.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('job' in dic.keys()):
            Str=Str+','+dic['job'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_job'])
train['crew_job']=tempDF['crew_job']
List=[]
for index,row in test.crew.iteritems():
    Str=''
    listofDict=ast.literal_eval(row)
    for dic in listofDict:
        
        if('job' in dic.keys()):
            Str=Str+','+dic['job'] 
    Str=Str.strip(',') # trim leading ,
    List.append(Str)
    
tempDF=pd.DataFrame(List,columns=['crew_job'])
test['crew_job']=tempDF['crew_job']
#bro= train['crew_department'].str.get_dummies(',')
#bro.drop(['Crew'],axis=1,inplace=True)
#train=train.join(bro)
#bro1=test['crew_department'].str.get_dummies(',')
#bro1.drop(['Crew'],axis=1,inplace=True)
#test=test.join(bro1)
#bro2=train['crew_job'].str.get_dummies(',')
#bro2columns= bro2.columns
#for i in bro2columns:
    #bro2.rename(columns={i: 'job'+i}, inplace=True)
#train=train.join(bro2)
#bro3=test['crew_job'].str.get_dummies(',')
#bro3columns= bro3.columns
#for i in bro3columns:
    #bro3.rename(columns={i: 'job'+i}, inplace=True)
#test=test.join(bro3)

dep=train['crew_department'].str.cat(sep=',')
words = dep.split(",")  
dep1  = Counter(words).most_common(10)
dep1
Dum=[]
for i in range(0,10):
  e=dep1[i][0]
  Dum.append(e)
for Dep in Dum:
    train['Department_' + Dep] = train['crew_department'].apply(lambda x: 1 if Dep in x else 0)
    test['Department_' + Dep] = test['crew_department'].apply(lambda x: 1 if Dep in x else 0)


Job=train['crew_job'].str.cat(sep=',')
words = Job.split(",")  
Job1  = Counter(words).most_common(15)
Job1
Dum=[]
for i in range(0,15):
  e=Job1[i][0]
  Dum.append(e)
for job in Dum:
    train['Job_' + job] = train['crew_job'].apply(lambda x: 1 if job in x else 0)
    test['Job_' + job] = test['crew_job'].apply(lambda x: 1 if job in x else 0)


train.belongs_to_collection=train.belongs_to_collection.fillna('[{}]')

train.belongs_to_collection=train.belongs_to_collection.apply(lambda x: 1 if x != '[{}]' else 0 )
test.belongs_to_collection=test.belongs_to_collection.fillna('[{}]')

test.belongs_to_collection=test.belongs_to_collection.apply(lambda x: 1 if x != '[{}]' else 0 )
train.homepage=train.homepage.fillna('[{}]')

train.homepage=train.homepage.apply(lambda x: 1 if x != '[{}]' else 0 )

test.homepage=test.homepage.fillna('[{}]')

test.homepage=test.homepage.apply(lambda x: 1 if x != '[{}]' else 0 )
#Seperating the date, day and year column
train['release_date'] = pd.to_datetime(train['release_date'])
train.loc[train['release_date'].dt.year >= 2020, 'release_date'] -= pd.DateOffset(years=100)
train['Year'], train['Month'],train['Date'] = train['release_date'].dt.year, train['release_date'].dt.month,train['release_date'].dt.day

test = test[test['release_date'].notnull()]
test['release_date'] = pd.to_datetime(test['release_date'])
test.loc[test['release_date'].dt.year >= 2020, 'release_date'] -= pd.DateOffset(years=100)
test['Year'], test['Month'],test['Date'] = test['release_date'].dt.year, test['release_date'].dt.month,test['release_date'].dt.day

train['First_Quarter_Release']=train.Month.apply(lambda x: 1 if x <= 3 else 0)
train['Second_Quarter_Release']=train.Month.apply(lambda x: 1 if x <= 6 and x > 3 else 0)
train['Third_Quarter_Release']=train.Month.apply(lambda x: 1 if x <= 9 and x > 6 else 0)
train['Fourth_Quarter_Release']=train.Month.apply(lambda x: 1 if x <= 12 and x > 9 else 0)
test['First_Quarter_Release']=test.Month.apply(lambda x: 1 if x <= 3 else 0)
test['Second_Quarter_Release']=test.Month.apply(lambda x: 1 if x <= 6 and x > 3 else 0)
test['Third_Quarter_Release']=test.Month.apply(lambda x: 1 if x <= 9 and x > 6 else 0)
test['Fourth_Quarter_Release']=test.Month.apply(lambda x: 1 if x <= 12 and x > 9 else 0)
#This block of code has been copid from kaggle notebook by Andrew Lukyan

for col in ['title', 'tagline', 'overview', 'original_title']:
    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))
    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))
    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))
#plotting different features vs target values
Budget=train.budget.value_counts().to_frame()
Budget.rename(columns={'budget':'Value_Counts'},inplace=True)
Budget.index.name = 'Budget'
Budget.reset_index()
plt.figure(figsize=(12,8))
ax1  = plt.subplot(1,2,1)
sns.regplot(x="budget", y="revenue", data=train)

ax2  = plt.subplot(1,2,2)
sns.kdeplot(np.log1p(train['budget']), shade=True)
plt.legend()

plt.show()

#list(train.columns.values.tolist()) 
Genres=['genre_Drama','genre_Comedy','genre_Thriller','genre_Action','genre_Romance','genre_Crime','genre_Adventure','genre_Horror','genre_Science Fiction','genre_Family','genre_Fantasy','genre_Mystery','genre_Animation','genre_History','genre_Music']
sns.barplot(train['num_of_genres'],np.log1p(train['revenue']))
f, axes = plt.subplots(3, 5, figsize=(28, 14))
plt.suptitle('Barplot of genres vs. their count')
for i, e in enumerate(Genres):
    Gr=train.groupby([e],as_index=False)['id'].count()
    Gr.rename(columns={'id':'Count'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Count", ax=axes[i //5 ][i % 5 ])
    
    

f, axes = plt.subplots(3, 5, figsize=(28, 14))
plt.suptitle('Barplot of genres vs. revenue')
for i, e in enumerate(Genres):
    Gr=train.groupby([e],as_index=False)['revenue'].mean()
    Gr.rename(columns={'revenue':'Revenue_Mean'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Revenue_Mean", ax=axes[i //5 ][i % 5 ])
sns.violinplot(data=train, x='homepage', y = "revenue")
stopwords = set(STOPWORDS)
Overview= ' '.join(train['overview'].fillna('').values)
plt.figure(figsize =(13,13))
ov_wc = WordCloud(
    background_color='white',
    max_words=500,
    stopwords=stopwords
)

# generate the word cloud
ov_wc.generate(Overview)
plt.imshow(ov_wc)
plt.axis('off')
plt.show()
count, bin_edges = np.histogram(train['popularity'])

train['popularity'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Popularity vs Count")
plt.xlabel("Popularity")
plt.ylabel("Count")

plt.show()
train.plot(kind='scatter', x='popularity', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Popularity vs Revenue')
plt.xlabel('Popularity')
plt.ylabel('Revenue')

plt.show()
train['log_revenue']=np.log1p(train['revenue'])
train.plot(kind='scatter', x='popularity', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Popularity vs Revenue')
plt.xlabel('Popularity')
plt.ylabel('Revenue')

plt.show()
PC=train.groupby(['num_of_productioncompanies'],as_index=False)['id'].count()
PC.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=PC, x='num_of_productioncompanies', y = "Count")
train.plot(kind='scatter', x='num_of_productioncompanies', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Production Companies vs Revenue')
plt.xlabel('Number of Production Companies')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='num_of_productioncompanies', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Production Companies vs Revenue')
plt.xlabel('Number of Production Companies')
plt.ylabel('Revenue')

plt.show()
Companies=['Company_Warner Bros.','Company_Universal Pictures','Company_Paramount Pictures','Company_','Company_Twentieth Century Fox Film Corporation','Company_Columbia Pictures','Company_Metro-Goldwyn-Mayer (MGM)','Company_New Line Cinema','Company_Touchstone Pictures','Company_Walt Disney Pictures','Company_Columbia Pictures Corporation','Company_TriStar Pictures','Company_Relativity Media','Company_Canal+','Company_United Artists']
f, axes = plt.subplots(3, 5, figsize=(28, 14))
plt.suptitle('Barplot of companies vs. revenue')
for i, e in enumerate(Companies):
    Gr=train.groupby([e],as_index=False)['revenue'].mean()
    Gr.rename(columns={'revenue':'Revenue_Mean'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Revenue_Mean", ax=axes[i //5 ][i % 5 ])
Countries=['Country_United States of America','Country_United Kingdom','Country_France','Country_Germany','Country_Canada','Country_India','Country_Italy','Country_Japan','Country_Australia','Country_Russia','Country_Spain','Country_China','Country_Hong Kong','Country_Ireland','Country_']
f, axes = plt.subplots(3, 5, figsize=(28, 14))
plt.suptitle('Barplot of countries vs. revenue')
for i, e in enumerate(Countries):
    Gr=train.groupby([e],as_index=False)['revenue'].mean()
    Gr.rename(columns={'revenue':'Revenue_Mean'},inplace=True)
    sns.barplot(data=Gr, x=e, y = "Revenue_Mean", ax=axes[i //5 ][i % 5 ])
SL=train.groupby(['num_of_spokenlanguages'],as_index=False)['id'].count()
SL.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=SL, x='num_of_spokenlanguages', y = "Count")
train.plot(kind='scatter', x='num_of_spokenlanguages', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Spoken Languages vs Revenue')
plt.xlabel('Number of Spoken Languages')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='num_of_spokenlanguages', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Spoken Languages vs Revenue')
plt.xlabel('Number of Spoken Languages')
plt.ylabel('Revenue')

plt.show()
Spoken_Languages=['Language_English','Language_Français','Language_Español','Language_Deutsch','Language_Pусский','Language_Italiano','Language_日本語','Language_普通话','Language_हिन्दी','Language_Português','Language_العربية','Language_한국어/조선말','Language_广州话 / 廣州話','Language_','Language_தமிழ்']
#f, axes = plt.subplots(3, 5, figsize=(28, 14))
#plt.suptitle('Barplot of spoken languages vs. revenue')
#for i, e in enumerate(Spoken_Languages):
    #Gr=train.groupby([e],as_index=False)['revenue'].mean()
    #Gr.rename(columns={'revenue':'Revenue_Mean'},inplace=True)
    #sns.barplot(data=Gr, x=e, y = "Revenue_Mean", ax=axes[i //5 ][i % 5 ])
keywords= ' '.join(train['Keywords'].fillna('').values)
plt.figure(figsize =(13,13))
kw_wc = WordCloud(
    background_color='white',
    max_words=500,
    stopwords=stopwords
)

# generate the word cloud
kw_wc.generate(keywords)
plt.imshow(kw_wc)
plt.axis('off')
plt.show()
train.plot(kind='scatter', x='num_of_keywords', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of keywords vs Revenue')
plt.xlabel('Number of Keywords')
plt.ylabel('Revenue')

plt.show()
avg_runtime = train['runtime'].astype('float').mean(axis=0)
train['runtime'].replace(np.nan, avg_runtime, inplace=True)
count, bin_edges = np.histogram(train['runtime'])

train['runtime'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Runtime vs Count")
plt.xlabel("Runtime")
plt.ylabel("Count")

plt.show()
train.plot(kind='scatter', x='runtime', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Runtime vs Revenue')
plt.xlabel('Runtime')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='runtime', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Runtime vs Revenue')
plt.xlabel('Runtime')
plt.ylabel('Revenue')

plt.show()
ST=train.groupby(['status'],as_index=False)['id'].count()
ST.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=ST, x='status', y = "Count")
plt.figure(figsize=(30,15))
CM=train.groupby(['total_castmembers'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='total_castmembers', y = "revenue_mean")
train.plot(kind='scatter', x='total_castmembers', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Total Cast Members vs Revenue')
plt.xlabel('Total Cast Members')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='total_castmembers', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Total Cast Members vs Revenue')
plt.xlabel('Total Cast Members')
plt.ylabel('Revenue')

plt.show()
plt.figure(figsize=(30,15))
CM=train.groupby(['total_crew_members'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='total_crew_members', y = "revenue_mean")
count, bin_edges = np.histogram(train['total_crew_members'])

train['total_crew_members'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Total crew members vs Count")
plt.xlabel("Total Crew Members")
plt.ylabel("Count")

plt.show()
train.plot(kind='scatter', x='total_crew_members', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Total Crew Members vs Revenue')
plt.xlabel('Total Crew Members')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='total_crew_members', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Total Crew Members vs Revenue')
plt.xlabel('Total Crew Members')
plt.ylabel('Revenue')

plt.show()
train.columns
plt.figure(figsize=(30,15))
CM=train.groupby(['Year'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='Year', y = "revenue_mean")
count, bin_edges = np.histogram(train['Year'])

train['Year'].plot(kind='hist', figsize=(8, 5), xticks=bin_edges)
plt.title("Year vs Count")
plt.xlabel("Year")
plt.ylabel("Count")

plt.show()
train.plot(kind='scatter', x='Year', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Year vs Revenue')
plt.xlabel('Year')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='Year', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Year vs Revenue')
plt.xlabel('Year')
plt.ylabel('Revenue')

plt.show()
plt.figure(figsize=(15,8))
CM=train.groupby(['Month'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='Month', y = "revenue_mean")
plt.figure(figsize=(15,8))
CM=train.groupby(['Month'],as_index=False)['id'].count()
CM.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=CM, x='Month', y = "Count")
train.plot(kind='scatter', x='Month', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Month vs Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='Month', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Month vs Revenue')
plt.xlabel('Month')
plt.ylabel('Revenue')

plt.show()
plt.figure(figsize=(15,8))
CM=train.groupby(['Date'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='Date', y = "revenue_mean")
plt.figure(figsize=(15,8))
CM=train.groupby(['Date'],as_index=False)['id'].count()
CM.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=CM, x='Date', y = "Count")
train.plot(kind='scatter', x='Date', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Date vs Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='Date', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Date vs Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')

plt.show()
train.columns
plt.figure(figsize=(12,6))
CM=train.groupby(['First_Quarter_Release'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='First_Quarter_Release', y = "revenue_mean")
plt.figure(figsize=(12,6))
CM=train.groupby(['First_Quarter_Release'],as_index=False)['id'].count()
CM.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=CM, x='First_Quarter_Release', y = "Count")
plt.figure(figsize=(12,6))
CM=train.groupby(['Second_Quarter_Release'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='Second_Quarter_Release', y = "revenue_mean")
plt.figure(figsize=(12,6))
CM=train.groupby(['Second_Quarter_Release'],as_index=False)['id'].count()
CM.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=CM, x='Second_Quarter_Release', y = "Count")
plt.figure(figsize=(12,6))
CM=train.groupby(['Third_Quarter_Release'],as_index=False)['revenue'].mean()
CM.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=CM, x='Third_Quarter_Release', y = "revenue_mean")
plt.figure(figsize=(12,6))
CM=train.groupby(['Third_Quarter_Release'],as_index=False)['id'].count()
CM.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=CM, x='Third_Quarter_Release', y = "Count")
plt.figure(figsize=(12,6))
FOQ=train.groupby(['Fourth_Quarter_Release'],as_index=False)['revenue'].mean()
FOQ.rename(columns={'revenue':'revenue_mean'},inplace=True)
sns.barplot(data=FOQ, x='Fourth_Quarter_Release', y = "revenue_mean")
plt.figure(figsize=(12,6))
FOQ=train.groupby(['Fourth_Quarter_Release'],as_index=False)['id'].count()
FOQ.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=FOQ, x='Fourth_Quarter_Release', y = "Count")
PCO=train.groupby(['num_of_productioncountries'],as_index=False)['id'].count()
PCO.rename(columns={'id':'Count'},inplace=True)
sns.barplot(data=PCO, x='num_of_productioncountries', y = "Count")
train.plot(kind='scatter', x='num_of_productioncountries', y='revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Production Countries vs Revenue')
plt.xlabel('Number of Production Countries')
plt.ylabel('Revenue')

plt.show()
train.plot(kind='scatter', x='num_of_productioncountries', y='log_revenue', figsize=(10, 6), color='darkblue')

plt.title('Number of Production Countries vs Revenue')
plt.xlabel('Number of Production Countries')
plt.ylabel('Revenue')

plt.show()
train.shape,test.shape
#list(train.columns.values.tolist()) 
# Converting big budget values into smaller values by taking log
train['log_budget']=np.log1p(train['budget'])
#Dropping the unnecessary columns
X=train.drop(['id','genres','imdb_id','original_language','original_title','overview','poster_path','production_companies','production_countries','release_date','spoken_languages','status', 'tagline', 'title', 'Keywords', 'cast', 'crew','revenue','crew_members','crew_department','crew_job','log_revenue','budget','Country_','Company_','Language_','Keyword_'],axis=1)
Y= train['log_revenue']
test['log_budget']=np.log1p(test['budget'])
test.drop(['id','genres','imdb_id','original_language','original_title','overview','poster_path','production_companies','production_countries','release_date','spoken_languages','status', 'tagline', 'title', 'Keywords', 'cast', 'crew','budget','crew_members','crew_department','crew_job','Country_','Company_','Language_','Keyword_'],axis=1,inplace =True)
X.shape,test.shape
#Splitting the dataset into train and test set
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,random_state=1,test_size=0.2)

from numpy import mean
from numpy import std
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# evaluate the model
model = GradientBoostingRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
model = GradientBoostingRegressor()
model.fit(train_X,train_Y )

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
mse = mean_squared_error(test_Y, model.predict(test_X))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
rmse = mean_squared_error(test_Y, model.predict(test_X),squared=False)
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
r2 = r2_score(test_Y, model.predict(test_X))
print("The R2  on test set: {:.4f}".format(r2))
import xgboost as xgb
from numpy import mean
from numpy import std
from numpy import asarray
from xgboost import XGBRegressor
# evaluate the model
modelXGB = XGBRegressor(objective='reg:squarederror')
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelXGB, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
modelXGB = XGBRegressor()

params = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4]}
grid = GridSearchCV(modelXGB, params)
grid.fit(train_X, train_Y)
best_model=grid.best_estimator_
msexgb = mean_squared_error(test_Y, best_model.predict(test_X))
print("The mean squared error (MSE) on test set: {:.4f}".format(msexgb))
rmsexgb = mean_squared_error(test_Y, best_model.predict(test_X),squared= False)
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmsexgb))
r2 = r2_score(test_Y, best_model.predict(test_X))
print("The R2  on test set: {:.4f}".format(r2))
fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(best_model, num_trees=1, ax=ax)
plt.show()

from lightgbm import LGBMRegressor
modellgbm = LGBMRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modellgbm, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
modellgbm = LGBMRegressor()
param_grid = {
        'objective': ['regression'],
        'num_leaves': [15, 23, 31],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100]}


gs_reg = GridSearchCV(modellgbm, param_grid,
                          n_jobs=1, cv=5,
                          scoring='neg_mean_squared_error')     
gs_reg.fit(train_X, train_Y)
best_lgbm = gs_reg.best_estimator_
mselgbm = mean_squared_error(test_Y, best_lgbm.predict(test_X))
print("The mean squared error (MSE) on test set: {:.4f}".format(mselgbm))
rmselgbm = mean_squared_error(test_Y, best_lgbm.predict(test_X),squared=False)
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmselgbm))
r2 = r2_score(test_Y, best_lgbm.predict(test_X))
print("The R2  on test set: {:.4f}".format(r2))
!pip install shap
#Trying to explain the results through shap. This is just an intro as I don't have enough knowledge about shap
import shap
%time shap_values = shap.TreeExplainer(best_lgbm).shap_values(test_X)
shap.summary_plot(shap_values, test_X)
shap.dependence_plot("log_budget", shap_values, test_X)
#Similarly visualize for other features

#!pip install catboost
from catboost import CatBoostRegressor
modelcat = CatBoostRegressor(verbose=0, n_estimators=100)
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(modelcat, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
# fit the model on the whole dataset
modelcat = CatBoostRegressor(iterations=100000,
                                 learning_rate=0.004,
                                 depth=5,
                                 eval_metric='RMSE',
                                 colsample_bylevel=0.8,
                                 random_seed = 2019,
                                 bagging_temperature = 0.2,
                                 metric_period = None,
                                 early_stopping_rounds=200
                                )
modelcat.fit(train_X, train_Y,
                 eval_set=(test_X,test_Y),
                 use_best_model=True,
                 verbose=False)
msecat = mean_squared_error(test_Y, modelcat.predict(test_X))
print("The mean squared error (MSE) on test set: {:.4f}".format(msecat))
rmsecat = mean_squared_error(test_Y, modelcat.predict(test_X),squared=False)
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmsecat))
r2 = r2_score(test_Y, modelcat.predict(test_X))
print("The R2  on test set: {:.4f}".format(r2))
#Making predictions on validation set and comparing it with target values
eval=modelcat.predict(test_X)
evaluations=np.expm1(eval)
evaluations1=np.expm1(test_Y)
Evaluations=pd.DataFrame()
Evaluations['revenue']=evaluations
Evaluations['original_revenue']=evaluations1

test.replace(np.nan, 0, inplace=True)
#making predictions on unknown test set
result=modelcat.predict(test)
#Taking back anti log
predictions=np.expm1(result)
predictions
Predictions=pd.DataFrame()
Predictions['revenue']=predictions
Predictions.head()
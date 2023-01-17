import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from wordcloud import WordCloud
from operator import itemgetter
data = pd.read_csv('../input/tmdb_5000_movies.csv')
data.info()
data.head()
data.columns
data.drop(['homepage','overview','original_language','production_countries','tagline','status','original_title','production_companies','spoken_languages'], axis = 1, inplace = True)
data.tail()
i = 0
result = []
df = pd.DataFrame()

while i < len(data.runtime):
    if data.runtime[i] > 200:
        result.append(data.title[i])
    i = i + 1

df['title'] = result
df
data.corr()
data.plot(kind='scatter', x='popularity', y='vote_count',alpha = 0.5,color = 'red',figsize = (10,10))
plt.show()
list1 = [] #list of dictionaries
list2 = [] #dictionaries
list3 = [] #list of frequencies of each category
counter = 0 
i = 0 
wordcloud = WordCloud(width=800, height=400)

for each in data.genres:
    list1.append(ast.literal_eval(each))

for each in list1:
    for each2 in each:
        list2.append(each2)

newlist = sorted(list2, key=lambda k: k['id'])

while i < len(newlist)-1:
    if newlist[i].get('id') == newlist[i+1].get('id'):
        counter += 1
        i += 1
    else:
        list3.append([newlist[i].get('name'),counter])
        counter = 0
        i += 1
        
#for each in list2:
#    list3.append(each['id'])

list3

df1 = pd.DataFrame(list3, columns = ['genre', 'freq'])

d = {}
for a, x in df1.values:
    d[a] = x


wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize=(20,10) )
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
data.runtime.plot(kind = 'hist',bins =40,figsize = (10,10))
plt.xlabel('Runtime (Minutes)')
plt.show()
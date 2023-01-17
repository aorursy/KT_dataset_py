import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#指定style以及字体

plt.style.use('ggplot')

sns.set(style='ticks',font='SimHei')
books=pd.read_csv('../input/goodreadsbooks/books.csv',error_bad_lines=False)

books.head(5)
books['authors'].replace('J.K. Rowling-Mary GrandPré','J.K. Rowling',inplace=True)

books.head()
print('共有{0}本书参与统计，其中不重复书籍为{1}本'.format(books.shape[0],books['title'].nunique()))
books_title20=books['title'].value_counts()[:20]



fig,ax=plt.subplots(figsize=(17,12))

ax=sns.barplot(x=books_title20,y=books_title20.index,palette='deep')

ax.set_xlabel('count')

plt.show()
books_lan=books['language_code'].value_counts()



fig,ax=plt.subplots(figsize=(17,5))

ax=sns.barplot(x=np.arange(30),y=books_lan,palette='deep')



ax.set_xticks(np.arange(30))

ax.set_xticklabels(books_lan.index,rotation=60)



ax.set_yticks(np.arange(0,12001,2000))

ax.set_ylabel('language_count')



for a,b in zip(np.arange(30),books_lan):

    ax.text(x=a,y=b,s='%.0f'%b,ha='center',va='bottom',fontsize=12,color='k')

    

plt.show()
books_rate=books.sort_values('ratings_count',ascending=False).head(10)

fig,ax=plt.subplots(figsize=(17,12))

ax=sns.barplot(books_rate['ratings_count'],books_rate['title'],palette='rocket')



ax.set_ylabel('')



plt.show()
books_author=books['authors'].value_counts().head(10)



fig,ax=plt.subplots(figsize=(15,10))

ax=sns.barplot(books_author,books_author.index,palette='icefire_r')



ax.set_xlabel('')



ax.set_ylabel('')



for c,d in zip(books_author,np.arange(10)):

    ax.text(x=c+1.5,y=d+0.2,s='%.0f'%c,ha='center',va='bottom',fontsize=18)

plt.show()
fig,ax=plt.subplots(figsize=(10,6))

ax=sns.distplot(books.average_rating,bins=20,color='r')



plt.show()
author_rate=pd.DataFrame()

author_rate['rate']=books['average_rating']

author_rate['group']=pd.cut(author_rate['rate'],

                            bins=np.arange(6),

                            labels=['0-1','1-2','2-3','3-4','4-5'])

author_rate=author_rate['group'].value_counts()

persent=author_rate/author_rate.sum()*100



fig,ax=plt.subplots(figsize=(10,10))

ax=plt.pie(author_rate,

           startangle=90)



plt.legend(['{0} : {1:1.2f} %'.format(e,f) for e,f in zip(author_rate.index,persent)],

          loc='upper right',

          bbox_to_anchor=(1,0,0.4,1))



plt.show()
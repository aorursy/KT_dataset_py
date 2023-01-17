import pandas as pn
import re

from string import ascii_lowercase
from datetime import datetime
#load data into dataframe
raw_books = pn.read_json('../input/nyt2.json', lines=True, orient='columns')
raw_books.head()
regex = re.compile('[^a-z0-9]')

#convert string to lowercase and replace all non alphanumeric characters
def set_lc_values(col_name):
    new_values = []
    for values in raw_books[col_name]:
        new_values.append(regex.sub('', values.lower()))

    return new_values



def flatten_json(y):
    # extract values from json string in dataframe
    
    out = {}
    ret_val = '-'
    
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        elif type(x) is str:
            out[name[:-1]] = x


    flatten(y)

    if type(out) is dict:
        for v in out:
            ret_val = out[v]
    
    return ret_val

#prepare data, extract values from json string, convert from unix stamp to datetime

for i in range(len(raw_books)):
    raw_books.loc[i]["_id"] = flatten_json(raw_books.loc[i]["_id"])
    raw_books.loc[i]["bestsellers_date"] = datetime.fromtimestamp(int(flatten_json(raw_books.loc[i]["bestsellers_date"])[:10])).strftime('%Y-%m-%d')
    raw_books.loc[i]["price"] = flatten_json(raw_books.loc[i]["price"])
    raw_books.loc[i]["published_date"] = datetime.fromtimestamp(int(flatten_json(raw_books.loc[i]["published_date"])[:10])).strftime('%Y-%m-%d')
    raw_books.loc[i]["rank"] = flatten_json(raw_books.loc[i]["rank"])
    raw_books.loc[i]["rank_last_week"] = flatten_json(raw_books.loc[i]["rank_last_week"])
    raw_books.loc[i]["title"] = flatten_json(raw_books.loc[i]["title"]) 
    raw_books.loc[i]["weeks_on_list"] = flatten_json(raw_books.loc[i]["weeks_on_list"])     
    
raw_books.head()
# add three new columns only with alphanumeric values for comparision and spellcheck detection

raw_books['l_author'] = set_lc_values('author')
raw_books['l_publisher'] = set_lc_values('publisher')
raw_books['l_title'] = set_lc_values('title')

raw_books.head()
print('Total number of titles: %s' % (len(raw_books['title'].unique())))
print('Total number of clean titles: %s' % (len(raw_books['l_title'].unique())))
#find just one clean lower title in data that have two different real titles
spec_title = raw_books.groupby(['l_title', 'title'])

doubleTitle = ''
for name, group in spec_title:
    if doubleTitle == name[0]:
        print('lower title: %s' % (name[0]))
    doubleTitle = name[0]

#find real titles based on single lower case title
filter_titles = raw_books[(raw_books['l_title'] == 'crossfire')].title.unique()
for t in filter_titles:
    print(raw_books[(raw_books['title'] == t)][['author', 'publisher', 'title']].reset_index(drop=True)[:1])
print('Total number of authors: %s' % (len(raw_books['author'].unique())))
print('Total number of clean authors: %s' % (len(raw_books['l_author'].unique())))
spec_author = raw_books.groupby(['l_author', 'author'])

double_author = ''
for name, group in spec_author:
    if double_author == name[0]:
        print('lower author: %s' % (name[0]))
    double_author = name[0]
#find real author name based on single lower case author name
filter_auth_col = raw_books[(raw_books['l_author'] == 'baparis')].author.unique()
for t in filter_auth_col:
    print(raw_books[(raw_books['author'] == t)][['author', 'publisher', 'title']].reset_index(drop=True)[:1])
print('Total number of publisher names: %s' % (len(raw_books['publisher'].unique())))
print('Total number of clean publisher names: %s' % (len(raw_books['l_publisher'].unique())))
spec_publisher = raw_books.groupby(['l_publisher', 'publisher'])

double_publisher = ''
for name,group in spec_publisher:
    if double_publisher == name[0]:
        print(name[0])
    double_publisher = name[0]
filter_auth_col = raw_books[(raw_books['l_publisher'] == 'stmartins')].publisher.unique()
for t in filter_auth_col:
    print(raw_books[(raw_books['publisher'] == t)][['publisher']].reset_index(drop=True)[:1])
print(raw_books['rank'].unique())
num_of_ranks = raw_books.groupby(['rank'])['rank'].count()
num_of_ranks
#date values
print(raw_books.bestsellers_date.min())
print(raw_books.bestsellers_date.max())
#e.g. top 5 books with number of weeks on rank #1
top_rank_books = raw_books[(raw_books['rank'] == '1')][['title', 'author', 'publisher']]
top_rank = top_rank_books.groupby(['title', 'author', 'publisher'])['title'].count().reset_index(name='weeks_rank_1')
top_rank.sort_values(by='weeks_rank_1', ascending=False)[:5]
#e.g. number of weeks that publisher have been on list as rank #1
top_rank_books = raw_books[(raw_books['rank'] == '1')][['publisher']]
top_rank = top_rank_books.groupby(['publisher'])['publisher'].count().reset_index(name='weeks_rank_1')
top_rank.sort_values(by='weeks_rank_1', ascending=False)[:5]
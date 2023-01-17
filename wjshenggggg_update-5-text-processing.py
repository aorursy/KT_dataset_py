import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.simplefilter('ignore')



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity



import re

import string 

from collections import Counter

from nltk.corpus import stopwords

stop = stopwords.words('english')



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.plotly as py

from plotly import tools

init_notebook_mode(connected=True)
emails = pd.read_csv('../input/emails.csv')

questions = pd.read_csv('../input/questions.csv')

professionals = pd.read_csv('../input/professionals.csv')

comments = pd.read_csv('../input/comments.csv')

tag_users = pd.read_csv('../input/tag_users.csv')

group_memberships = pd.read_csv('../input/group_memberships.csv')

tags = pd.read_csv('../input/tags.csv')

students = pd.read_csv('../input/students.csv')

groups = pd.read_csv('../input/groups.csv')

tag_questions = pd.read_csv('../input/tag_questions.csv')

matches = pd.read_csv('../input/matches.csv')

answers = pd.read_csv('../input/answers.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')
def merging(df1, df2, left, right):

    return df1.merge(df2, how="inner", left_on=left, right_on=right)
qa = merging(questions, answers, "questions_id", "answers_question_id")

qa.head(3).T
qa['questions_date_added'] = pd.to_datetime(qa['questions_date_added'])

qa['answers_date_added'] = pd.to_datetime(qa['answers_date_added'])

qa['qa_duration'] = (qa['answers_date_added'] - qa['questions_date_added']).dt.days



qa.head().T
# after groupby, head(1) returns the first occurrence

first_qa = qa.groupby('questions_id').head(1)



# let's explore data from last year

first_qa = first_qa[first_qa['questions_date_added'] >= pd.datetime(2018, 1, 1)]



first_qa.loc[(first_qa['qa_duration'] <= 7), 'week'] = 1

first_qa.loc[(first_qa['qa_duration'] > 7) & (first_qa['qa_duration'] <= 14), 'week'] = 2

first_qa.loc[(first_qa['qa_duration'] > 14) & (first_qa['qa_duration'] <= 21), 'week'] = 3

first_qa.loc[(first_qa['qa_duration'] > 21) & (first_qa['qa_duration'] <= 28), 'week'] = 4

first_qa.loc[(first_qa['qa_duration'] > 28), 'week'] = 5
week_val_cnt = first_qa['week'].value_counts().sort_index()



plt.figure(figsize=(8,6))

sns.barplot(week_val_cnt.index, 

            week_val_cnt.values)



plt.xlabel('Week')

plt.ylabel('Responses')

plt.title('Responses vs Week')

plt.show()
def process_text(df, col):

    df[col] = df[col].str.replace('[^\w\s]','') # replacing punctuations

    df[col] = df[col].str.replace('-',' ') # replacing dashes

    df[col] = df[col].str.replace('\d+','') # replacing digits

    df[col] = df[col].str.lower().str.split() # convert all str to lowercase    

    df[col] = df[col].apply(lambda x: [item for item in x if item not in stop]) # remove stopwords    

    df[col] = df[col].apply(' '.join) # convert list to str

    return df
first_qa['questions_body'] = process_text(first_qa, 'questions_body')['questions_body']



fast_resp = pd.Series(first_qa[first_qa['week'] == 1]['questions_body'].tolist()).astype(str)

slow_resp = pd.Series(first_qa[first_qa['week'] == 5]['questions_body'].tolist()).astype(str)



dist_fast = fast_resp.apply(lambda x: len(x.split(' ')))

dist_slow = slow_resp.apply(lambda x: len(x.split(' ')))
pal = sns.color_palette()



plt.figure(figsize=(18, 8))

plt.hist(dist_fast, bins=40, range=[0, 80], color=pal[9], normed=True, label='fast')

plt.hist(dist_slow, bins=40, range=[0, 80], color=pal[1], normed=True, alpha=0.5, label='slow')

plt.title('Normalised histogram of word count in question_body', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)
from wordcloud import WordCloud



all_q = process_text(first_qa, 'questions_body')['questions_body']

cloud = WordCloud(width=1440, height=1080).generate(" ".join(all_q.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
tf = TfidfVectorizer(analyzer='word',

                     min_df=3,

                     max_df=0.9,

                     stop_words='english')



# generate a matrix of sentences and a score for each word

fast_tfidf_matrix = tf.fit_transform(fast_resp)



# generate a list of words from the vectorizer

fast_vocab = tf.get_feature_names()



# repeat for slow response

slow_tfidf_matrix = tf.fit_transform(slow_resp)

slow_vocab = tf.get_feature_names()
# sum of the scores of each word

# each row represents a sentence

# each column represents a word

# we have to sum across all columns



def word_score_pair(matrix, vocab):

    mat_to_arr = matrix.toarray() # convert the 2d matrix to a 2d array

    word_score = list(map(sum,zip(*mat_to_arr))) # fastest way to sum across all columns

    rank_words_idx = np.argsort(word_score)

    idx_list = rank_words_idx[:10]

    

    for idx in idx_list:

        print("word: {0}, score: {1:.3f}".format(vocab[idx], word_score[idx]))
print('fast_vocab\'s words and score:')

word_score_pair(fast_tfidf_matrix, fast_vocab)
print('slow_vocab\'s words and score:')

word_score_pair(slow_tfidf_matrix, slow_vocab)
state_codes = {'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 

               'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 

               'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 

               'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 

               'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 

               'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 

               'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 

               'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',

               'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 

               'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 

               'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 

               'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 

               'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 

               'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 

               'Nevada': 'NV', 'Maine': 'ME'}
students['students_location'] = students['students_location'].fillna('')

students['students_location'] = students['students_location'].str.split(',').str[1]

students['students_location'] = students['students_location'].str.lstrip() # remove first white space



s_val_cnt = students['students_location'].value_counts()

s_val_cnt[:10]
us_states = []



# only get the location if it's in US

for s in s_val_cnt.index.tolist():

    if s in state_codes:

        us_states.append(s)
df = pd.DataFrame({'states': s_val_cnt.index,

                   'count': s_val_cnt.values})



df = df[df['states'].isin(us_states)]

df['states'] = df['states'].apply(lambda x: state_codes[x])
data = [ dict(

        type='choropleth',

        autocolorscale = True,

        locations = df['states'], 

        z = df['count'].astype(float), 

        locationmode = 'USA-states', 

        text = df['states'], 

        marker = dict(

            line = dict (

                color = 'rgb(255,255,255)',

                width = 2

            ) ),

        colorbar = dict(  

            title = "count")  

        ) ]



layout = dict(

        title = 'Number of Students by State<br>(Hover for breakdown)',

        geo = dict(

            scope='usa',

            projection=dict( type='albers usa' ),

            showlakes = True,

            lakecolor = 'rgb(255, 255, 255)'),

             )



fig = dict(data=data, layout=layout)

iplot(fig)
qap = merging(qa, professionals, "answers_author_id", "professionals_id")

qap.head(3).T
p_industry_cnt = professionals['professionals_industry'].value_counts()



plt.figure(figsize=(10,8))

sns.barplot(p_industry_cnt.index, 

            p_industry_cnt.values,

            order=p_industry_cnt.iloc[:10].index)



plt.xticks(rotation=90)

plt.xlabel('professionals_industry', fontsize=16)

plt.ylabel('counts', fontsize=16)

plt.title('counts vs professionals_industry', fontsize=18)

plt.show()
p_cnt = professionals['professionals_headline'].value_counts()



plt.figure(figsize=(10,8))

sns.barplot(p_cnt.index, 

            p_cnt.values,

            order=p_cnt.iloc[1:11].index) # 1 to 11 because we remove NaNs



plt.xticks(rotation=90)

plt.xlabel('professionals_headline', fontsize=16)

plt.ylabel('counts', fontsize=16)

plt.title('counts vs professionals_headline', fontsize=18)

plt.show()
qap_author_id = qap['answers_author_id'].value_counts()



plt.figure(figsize=(10,8))

sns.barplot(qap_author_id.index, 

            qap_author_id.values,

            order=qap_author_id.iloc[:10].index)



plt.xticks(rotation=90)

plt.xlabel('answers_author_id', fontsize=16)

plt.ylabel('counts', fontsize=16)

plt.title('counts vs answers_author_id', fontsize=18)

plt.show()
p = professionals.copy()

p['professionals_location'] = p['professionals_location'].str.split(',').str[1]



p_cnt = p['professionals_location'].value_counts()



plt.figure(figsize=(10,8))

sns.barplot(p_cnt.index, 

            p_cnt.values,

            order=p_cnt.iloc[0:10].index) # 1 to 11 because we remove NaNs



plt.xticks(rotation=90)

plt.xlabel('professionals_location', fontsize=16)

plt.ylabel('counts', fontsize=16)

plt.title('counts vs professionals_location', fontsize=18)

plt.show()
pa = merging(professionals, answers, "professionals_id", "answers_author_id")



# get active authors

pa['ans_cnt'] = 1

p = pa.groupby('professionals_id')['ans_cnt'].sum()

active_p = (p[p > 5].index).tolist()

active_p



# get an updated list of authors

pa['answers_date_added'] = pd.to_datetime(pa['answers_date_added'])

recent_p = (pa[pa['answers_date_added'] >= pd.datetime(2018, 1, 1)]['professionals_id']).tolist()

recent_p



# get the intersection of both recent and active authors

active_recent_p = list(set(recent_p) & set(active_p))



len(recent_p), len(active_p), len(active_recent_p)
pa = merging(professionals, answers, "professionals_id", "answers_author_id")

pa.head().T
before = pa.iloc[0]['answers_body'][:496]

before
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def strip_html(s):

    return re.sub(uri_re, ' ', str(s))
pa['answers_body'] = pa['answers_body'].apply(strip_html)

pa['answers_body'] = pa['answers_body'].str.replace('[^\w\s\n\t]',' ') # replace punctuations

pa['answers_body'] = pa['answers_body'].str.lower().str.split() # convert all str to lowercase

pa['answers_body'] = pa['answers_body'].apply(lambda x: [item for item in x if item not in stop]) # remove stopwords

pa['answers_body'] = pa['answers_body'].apply(' '.join) # convert list to str
after = pa.iloc[0]['answers_body'][:496]

after
all_a = pa['answers_body']

cloud = WordCloud(width=1440, height=1080).generate(" ".join(all_a.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
ttq = merging(tags, tag_questions, "tags_tag_id", "tag_questions_tag_id")

qttq = merging(questions, ttq, "questions_id", "tag_questions_question_id")
tqq_list = ttq['tag_questions_question_id'].tolist()

questions.shape[0], questions[~questions['questions_id'].isin(tqq_list)].shape[0]
val_cnt = ttq['tags_tag_name'].value_counts()

to_replace = val_cnt[val_cnt <= 5].index.tolist()



print("Top 10 most popular tags:")

print(val_cnt[:10], '\n')

print("Number of unique tags: ", ttq['tags_tag_name'].nunique())

print("Number of tags that occur 5 times and below: ", len(to_replace))
top_10_val_cnt = val_cnt[:10]



fig = {

    "data": [

    {

      "values": top_10_val_cnt.values,

      "labels": top_10_val_cnt.index,

      "domain": {"x": [0, .48]},

      "marker" : dict(colors=["#f77b9c" ,'#ab97db',  '#b0b1b2']),

      "name": "tag count",

      "hoverinfo":"label+percent+name",

      "hole": .5,        

      "type": "pie"

    }],

    "layout": {

      "title":"Tags and Count",

      "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "Tags",

                "x": 0.2,

                "y": 0.5

            }]

    }

}

        

iplot(fig, filename='plot-0')
def search_pat(pat, tags_list):

    sim_pat = []

    for s in tags_list:

        if pat in s:

            sim_pat.append(s)    

    return sim_pat
tags_list = val_cnt.index.tolist()

c_idx = []

c_val = []



for c in search_pat("college", tags_list)[:10]:

    c_idx.append(c)

    c_val.append(val_cnt[c])



df = pd.DataFrame({'variation of #college': c_idx,

                   'counts': c_val})
fig = {

    "data": [

    {

      "values": df['counts'],

      "labels": df['variation of #college'],

      "domain": {"x": [0, .48]},

      "marker" : dict(colors=["#f77b9c","#efbc56", "#81a7e8", "#e295d0"]),

      "name": "count",

      "hoverinfo":"label+percent+name",

      "hole": .5,        

      "type": "pie"

    }],

    "layout": {

      "title":"#college and Count",

      "annotations": [

            {

                "font": {

                    "size": 20

                },

                "showarrow": False,

                "text": "#college",

                "x": 0.16,

                "y": 0.5

            }]

    }

}

        

iplot(fig, filename='plot-1')
def multi_single_tags(df, tag):

    without_tag = df[df['tags_tag_name'] != tag]['tag_questions_question_id'].tolist()

    with_tag = df[df['tags_tag_name'] == tag]['tag_questions_question_id'].tolist()

    

    only_tag = df[~df['tag_questions_question_id'].isin(without_tag)]['tag_questions_question_id'].tolist()



    multiple_tags = list(set(with_tag) - set(only_tag))

    

    return multiple_tags, only_tag
def remove_multiple(df, tag, ids_multiple, ids_single):

    df = df[((df['questions_id'].isin(ids_multiple)) & (df['tags_tag_name'] != tag)) | 

             (df['tags_tag_name'] != tag) |

             (df['questions_id'].isin(ids_single))]

    return df
college_ids_multiple, college_ids_single = multi_single_tags(ttq, "college")



print('Before removing multiple tags containing #college, we have {} questions.'.format(qttq.shape[0]))



qttq = remove_multiple(qttq, "college", college_ids_multiple, college_ids_single)



print('After removing multiple tags containing #college, we are left with {} questions.'.format(qttq.shape[0]))
def combine_tags(df):

    grouped = df.groupby('questions_id')['tags_tag_name'].apply(lambda x: "%s" % ', '.join(x))

    df_c = merging(questions, pd.DataFrame(grouped), "questions_id", "questions_id")

    return df_c
combine_qttq = combine_tags(qttq)

combine_qttq.head().T
# qapttq = merging(qap, ttq, "questions_id", "tag_questions_question_id")

qapttq = merging(answers, combine_qttq, "answers_question_id", "questions_id")

qapttq.head().T
qapttq.shape[0], combine_qttq.shape[0]
noise = ['school','would','like', 'want', 'dont', 

         'become','sure','go', 'get', 'college', 

         'career', 'wanted', 'im', 'ing', 'ive',

         'know', 'high', 'becom', 'job', 'best',

         'day', 'hi', 'name', 'help', 'people',

         'year', 'years', 'next', 'interested', 

         'question', 'questions', 'take', 'even',

         'though', 'please', 'tell']
def another_process_text(df, col):

    df[col] = df[col].str.replace('[^\w\s]','') # replacing punctuations

    df[col] = df[col].str.replace('-',' ') # replacing dashes

    df[col] = df[col].str.replace('\d+','') # replacing digits

    df[col] = df[col].str.lower().str.split() # convert all str to lowercase    

    df[col] = df[col].apply(lambda x: [item for item in x if item not in stop]) # remove stopwords

    df[col] = df[col].apply(lambda x: [item for item in x if item not in noise])

    df[col] = df[col].apply(' '.join) # convert list to str

    return df



def generate_ngrams(text, N):

    grams = [text[i:i+N] for i in range(len(text)-N+1)]

    grams = [" ".join(b) for b in grams]

    return grams
df = another_process_text(questions, 'questions_body')
df['bigrams'] = df['questions_body'].apply(lambda x : generate_ngrams(x.split(), 2))
all_bigrams = []



for each in df['bigrams']:

    all_bigrams.extend(each)

    

t1 = Counter(all_bigrams).most_common(20)

x1 = [a[0] for a in t1]

y1 = [a[1] for a in t1]
fig, axes = plt.subplots(figsize=(15,10))



bar = sns.barplot(y=x1, x=y1)

bar.set(ylabel='Most frequent bigrams', xlabel='Frequency')
def tag_chart(df, what_tag, top):

    """

    df: the DataFrame

    what_tag: tags we are looking for

    top: number of professionals in the chart after filtering

    """    

    mod_df = df[['answers_author_id', 'tags_tag_name']].copy()

    mod_df['tag_count'] = 1

    grouped = mod_df.groupby(['tags_tag_name', 'answers_author_id']).sum()

    grouped_df = (grouped.reset_index()

                         .sort_values(['tags_tag_name', 'tag_count'], ascending=False)

                         .set_index(['answers_author_id']))



    grouped_filter = grouped_df[grouped_df['tags_tag_name'] == what_tag]['tag_count'].reset_index()

    return grouped_filter.head(top)
# remerge our qapttq since we modified the tags in the previous one

qapttq = merging(qap, ttq, "questions_id", "tag_questions_question_id")



tag_chart(qapttq, "college", 5), tag_chart(qapttq, "engineering", 5) 
def combine_authors(df):

    c = df.groupby('questions_id')['answers_author_id'].apply(list)

    df_c = merging(df, pd.DataFrame(c), 'questions_id', 'questions_id')

    df_c.drop('answers_author_id_x', axis=1, inplace=True)

    df_c['answers_author_id_y'] = df_c['answers_author_id_y'].apply(', '.join)

    df_c.drop_duplicates(inplace=True)

    return df_c
qa_sub = qa[['questions_title', 'questions_body', 'answers_author_id', 'questions_id']].copy()



qa_cbr = combine_authors(qa_sub)



authors_link = qa_cbr[['questions_id', 'answers_author_id_y']].copy()



qa_cbr.drop('answers_author_id_y', axis=1, inplace=True)



qa_cbr.head()
# hacky way to remove authors who are not linked

authors_link = authors_link[authors_link['answers_author_id_y'].str.len() > 33] 



authors_link_dic = authors_link.set_index('questions_id').T.to_dict()
qa_cbr = process_text(qa_cbr, "questions_title") 

qa_cbr = process_text(qa_cbr, "questions_body") 



qa_cbr.head()
tf = TfidfVectorizer(analyzer='word',

                     ngram_range=(1,2),

                     min_df=3,

                     max_df=0.9,

                     stop_words='english')



tfidf_matrix = tf.fit_transform(qa_cbr['questions_body'])

tfidf_matrix.shape
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# qa_cbr = qa_cbr.reset_index()

q_titles = qa_cbr['questions_title']

q_ids = qa_cbr['questions_id']

indices = pd.Series(qa_cbr.index, index=qa_cbr['questions_title'])



qa_cbr.head()
def get_recommendations_idx(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    q_indices = [i[0] for i in sim_scores]

    return q_indices



def get_recommendations(title):

    return q_titles.iloc[get_recommendations_idx(title)]

    

def get_questions_id(title):

    return q_ids.iloc[get_recommendations_idx(title)]    
get_recommendations('want become army officer become army officer').head(10)
get_questions_id('want become army officer become army officer').head(10)
def get_sim_authors(qids):

    sim_authors = []

    for qid in qids:

        if qid in authors_link_dic:

            sim_authors.append(authors_link_dic[qid]['answers_author_id_y'])

    return sim_authors
qids = get_questions_id('want become army officer become army officer').tolist()



qa[qa['questions_id'].isin(qids)].head()
sim_ids = []

for all_ids in get_sim_authors(qids):

    for each_id in all_ids.split(','):

        sim_ids.append(each_id)



sim_ids = set(sim_ids)



sim_active_recent = set(active_recent_p) & set(sim_ids)



sim_active_recent



professionals[professionals['professionals_id'].isin(sim_active_recent)].T
from sklearn.manifold import TSNE



tsne = TSNE(random_state=0, n_iter=250, metric="cosine")
g_q_sample = qa_cbr.sample(frac=.4, random_state=43)



tf = TfidfVectorizer(analyzer='word',

                     ngram_range=(1,2),

                     min_df=0,

                     stop_words='english')



tfidf_matrix = tf.fit_transform(g_q_sample['questions_body'])

tfidf_matrix.shape



tm = tfidf_matrix.toarray()



tsne_matrix = tsne.fit_transform(tm)



tsne_matrix
df = g_q_sample.copy()



df['x'] = tsne_matrix[:, 0]

df['y'] = tsne_matrix[:, 1]
FS = (10, 8)

fig, ax = plt.subplots(figsize=FS)

# Make points translucent so we can visually identify regions with a high density of overlapping points

ax.scatter(df.x, df.y, alpha=.1)
FS = (18, 8)

def plot_region(x0, x1, y0, y1, text=True):

    """

    Plot the region of the mapping space bounded by the given x and y limits.

    """    

    pts = df[

        (df.x >= x0) & (df.x <= x1)

        & (df.y >= y0) & (df.y <= y1)

    ]

    fig, ax = plt.subplots(figsize=FS)

    ax.scatter(pts.x, pts.y, alpha=.6)

    ax.set_xlim(x0, x1)

    ax.set_ylim(y0, y1)

    if text:

        texts = []

        for label, x, y in zip(pts.questions_title.values, pts.x.values, pts.y.values):

            t = ax.annotate(label, xy=(x, y))

            texts.append(t)

    return ax



def plot_region_around(title, margin=5, **kwargs):

    """

    Plot the region of the mapping space in the neighbourhood of the the questions_title. 

    The margin parameter controls the size of the neighbourhood around the movie.

    """

    xmargin = ymargin = margin

    match = df[df.questions_title == title]

    assert len(match) == 1

    row = match.iloc[0]

    return plot_region(row.x-xmargin, row.x+xmargin, row.y-ymargin, row.y+ymargin, **kwargs)
# df

plot_region_around('lifestyle pediatric surgeon', .00005)
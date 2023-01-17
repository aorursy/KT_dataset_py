# The data comes as the raw data files, a transformed CSV file, and a SQLite database

from heapq import nlargest
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# You can read in the SQLite datbase like this
def remove_punctuation(text):
    import string
   
    return text.translate(string.punctuation) 

def count_number_papers(id1):
    '''papers_id = paper_authors['AuthorId']'''
    tam = len(paper_authors)
    count_works = 0;
    count_poster = 0;
    count_spotlight = 0;
    count_oral = 0;
    
    for i in range(0,tam):
        if id1 == paper_authors['AuthorId'][i]:
            count_works = count_works + 1;
            linha = papers[papers['Id'] == paper_authors['PaperId'][i]]
            if linha['EventType'].any() == 'Poster':
               count_poster = count_poster + 1
            if linha['EventType'].any() == 'Spotlight':
               count_spotlight = count_spotlight + 1
            if linha['EventType'].any() == 'Oral':
               count_oral = count_oral + 1   
    return(count_works,count_poster,count_oral,count_spotlight)

def abstract_by_author(paperID):
    df = pd.DataFrame()
    for i in range(0,len(paperID)):
        df = df.append(papers[papers['Id'] == paperID.iloc[i]])
     
    return df
    
def words_by_author(authorId):
    PaperId = paper_authors[paper_authors['AuthorId'] == authorId]
    PaperId = PaperId['PaperId']
    dado = abstract_by_author(PaperId)
    
    vectorizer = CountVectorizer(analyzer='word',decode_error='strict', encoding='utf-8' ,input='content',  token_pattern=r'\b\w+\b', stop_words='english')
    
   
    train_matrix1 = vectorizer.fit_transform(dado['Abstract_clean']).toarray()
    train_matrix1 = train_matrix1.sum(axis=0)
    vocab = vectorizer.get_feature_names()
    posi = nlargest(10,range(len(train_matrix1[:])), train_matrix1[:].__getitem__)
    tel = dict()
    for i in enumerate(posi):
        
        index = i[1]
        tel[vocab[index]] = train_matrix1[i[1]]
    return(tel)


def words_by_author_bigram(authorId):
    PaperId = paper_authors[paper_authors['AuthorId'] == authorId]
    PaperId = PaperId['PaperId']
    dado = abstract_by_author(PaperId)
    
    
    vectorizer = CountVectorizer(ngram_range=(2,2), token_pattern=r'\b\w+\b', min_df=1, stop_words='english')
    train_matrix1 = vectorizer.fit_transform(dado['Abstract_clean']).toarray()
    
    train_matrix1 = train_matrix1.sum(axis=0)
    vocab = vectorizer.get_feature_names()
    
    posi = nlargest(10,range(len(train_matrix1[:])), train_matrix1[:].__getitem__)
    tel = dict()
    for i in enumerate(posi):
        
        index = i[1]
        tel[vocab[index]] = train_matrix1[i[1]]
    return(tel)    
    
papers = pd.read_csv("../input/Papers.csv", sep=',', encoding='latin1')
paper_authors = pd.read_csv("../input/PaperAuthors.csv", sep=',', encoding='latin1')
authors = pd.read_csv("../input/Authors.csv", sep=',', encoding='latin1')

authors['number_works'], authors['type_poster'], authors['type_oral'],authors['type_spotlight'] = zip(*authors['Id'].apply(count_number_papers))
authors = authors.sort_values(['number_works'], ascending=False)

papers['Abstract'].fillna('',)
papers['Abstract_clean'] = papers['Abstract'].apply(remove_punctuation)

authors['most_words_unigram'] = authors['Id'].apply(words_by_author)
authors['most_words_bigram'] = authors['Id'].apply(words_by_author_bigram)

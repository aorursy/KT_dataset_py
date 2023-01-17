import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import json
from tqdm import tqdm_notebook
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()
PATH_TO_DATA = '../input/'
!ls -l $PATH_TO_DATA
def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result
def preprocess(path_to_inp_json_file):
    output_list = []
    published_list = [] 
    title_list = []
    author_list = []
    domain_list = []
    tags_list = []
    url_list = []
    with open(path_to_inp_json_file, encoding='utf-8') as inp_file:
        for line in tqdm_notebook(inp_file):
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' ').replace('\r', ' ')
            content_no_html_tags = strip_tags(content)
            output_list.append(content_no_html_tags)
            published = json_data['published']['$date']
            published_list.append(published) 
            title = json_data['meta_tags']['title'].split('\u2013')[0].strip() #'Medium Terms of Service – Medium Policy – Medium'
            title_list.append(title) 
            author = json_data['meta_tags']['author'].strip()
            author_list.append(author) 
            domain = json_data['domain']
            domain_list.append(domain)
            url = json_data['url']
            url_list.append(url)
    return (output_list,published_list,title_list,author_list,domain_list,url_list)
%%time
raw_content_features = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 
                                                                  'train.json'),)
%%time
#train_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 
                                                                 # 'train.json'),)
raw_content_features_test = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 
                                                                  'test.json'),)
#%%time
#test_raw_content = preprocess(path_to_inp_json_file=os.path.join(PATH_TO_DATA, 
                                                                 # 'test.json'),)
df_full_1=pd.DataFrame()
df_full_1["published"] = pd.to_datetime(raw_content_features_test[1], format='%Y-%m-%dT%H:%M:%S.%fZ')
df_full_1['dow'] = df_full_1['published'].apply(lambda x: x.dayofweek)
df_full_1['year'] = df_full_1['published'].apply(lambda x: x.year)
df_full_1['month'] = df_full_1['published'].apply(lambda x: x.month)
df_full_1['hour'] = df_full_1['published'].apply(lambda x: x.hour)
df_full=pd.DataFrame()
df_full["published"] = pd.to_datetime(raw_content_features[1], format='%Y-%m-%dT%H:%M:%S.%fZ')
df_full['dow'] = df_full['published'].apply(lambda x: x.dayofweek)
df_full['year'] = df_full['published'].apply(lambda x: x.year)
df_full['month'] = df_full['published'].apply(lambda x: x.month)
df_full['hour'] = df_full['published'].apply(lambda x: x.hour)

df_full.columns
X_test_time_features_sparse=pd.get_dummies(df_full_1[['dow', 'year', 'month', 'hour']])
X_train_time_features_sparse=pd.get_dummies(df_full[['dow', 'year', 'month', 'hour']])
cv = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
%%time
X_train_content_sparse = cv.fit_transform(raw_content_features[0])
%%time
X_test_content_sparse = cv.fit_transform(raw_content_features_test[0])
%%time
X_train_title_sparse = cv.fit_transform(raw_content_features[2])
%%time
X_test_title_sparse = cv.fit_transform(raw_content_features_test[2])
author=raw_content_features[3] +raw_content_features_test[3]
len(author)
Cvt=CountVectorizer().fit(author)
%%time
X_train_author_sparse = Cvt.transform(raw_content_features[3])
%%time
X_test_author_sparse = Cvt.transform(raw_content_features_test[3])

X_train_author_sparse.shape
X_test_author_sparse.shape
%%time
#X_test_author_sparse = CountVectorizer().fit_transform(raw_content_features_test[3])
X_train_sparse = hstack([X_train_content_sparse, X_train_title_sparse,
                         X_train_author_sparse,X_train_time_features_sparse]).tocsr()

X_test_sparse = hstack([X_test_content_sparse, X_test_title_sparse,
                         X_test_author_sparse,X_test_time_features_sparse]).tocsr()
X_train_sparse.shape,X_test_sparse.shape
X_train=X_train_sparse
X_test=X_test_sparse
train_target = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_log1p_recommends.csv'), 
                           index_col='id')
train_target.shape
y_train = train_target['log_recommends'].values
train_part_size = int(0.7 * train_target.shape[0])
X_train_part = X_train[:train_part_size, :]
y_train_part = y_train[:train_part_size]
X_valid =  X_train[train_part_size:, :]
y_valid = y_train[train_part_size:]
from sklearn.linear_model import Ridge
ridge=Ridge(random_state=17)
    
%%time
ridge.fit(X_train_part, y_train_part);
ridge_pred = ridge.predict(X_valid)
plt.hist(y_valid, bins=30, alpha=.5, color='red', label='true', range=(0,10));
plt.hist(ridge_pred, bins=30, alpha=.5, color='green', label='pred', range=(0,10));
plt.legend();
valid_mae = mean_absolute_error(y_valid, ridge_pred)
valid_mae, np.expm1(valid_mae)
%%time
ridge.fit(X_train, y_train);
%%time
ridge_test_pred = ridge.predict(X_test)
def write_submission_file(prediction, filename,
    path_to_sample=os.path.join(PATH_TO_DATA, 'sample_submission.csv')):
    submission = pd.read_csv(path_to_sample, index_col='id')
    
    submission['log_recommends'] = prediction
    submission.to_csv(filename)
write_submission_file(prediction=ridge_test_pred, 
                      filename='first_ridge.csv')
write_submission_file(ridge_test_pred, 'assignment6_medium_submission.csv')
write_submission_file(np.zeros_like(ridge_test_pred),
                                   'medium_all_zeros_submission.csv')
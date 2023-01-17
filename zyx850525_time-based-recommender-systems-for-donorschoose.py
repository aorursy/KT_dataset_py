from datetime import datetime
import gc
from multiprocessing import Pool
import random
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
debug = True
df_donations = pd.read_csv('../input/Donations.csv', 
                           dtype={'Donation Amount': 'float32', 'Donor Cart Sequence': 'uint16'}, 
                           index_col='Donation ID')


# 'Donor ID' can be used as index
df_donors = pd.read_csv('../input/Donors.csv')


# 'Project ID' can be used as index after delete 2 records
df_projects = pd.read_csv('../input/Projects.csv',        
                          dtype={'Teacher Project Posted Sequence': 'uint16', 'Project Cost': 'float32'})
df_projects.drop_duplicates('Project ID', keep='first', inplace=True)
donotion_cnt = len(df_donations.index)
donation_donor_cnt = df_donations['Donor ID'].nunique()
donation_project_cnt = df_donations['Project ID'].nunique()
donor_cnt = df_donors['Donor ID'].nunique()
project_cnt = df_projects['Project ID'].nunique()

print('# %d unique donations in donations file' %donotion_cnt)
print('# %d unique donors in donations file' %donation_donor_cnt)
print('# %d unique projects in donations file' %donation_project_cnt)
print('# %d unique donors in donors file' %donor_cnt)
print('# %d unique projects in projects file' %project_cnt)
diff_project = set(df_donations['Project ID'].unique()) - set(df_projects['Project ID'].unique())
diff_donor = set(df_donations['Donor ID'].unique()) - set(df_donors['Donor ID'].unique())
print('# %d projects in donations are not included in projects' %len(diff_project))
print('# %d donors in donations are not included in donors' %len(diff_donor))

print('-'*60)

diff_project = set(df_projects['Project ID'].unique()) - set(df_donations['Project ID'].unique())
diff_donor = set(df_donors['Donor ID'].unique()) - set(df_donations['Donor ID'].unique())
print('# %d projects in projects are not included in donations' %len(diff_project))
print('# %d donors in donors are not included in donations' %len(diff_donor))
del donotion_cnt, donation_donor_cnt, donation_project_cnt, donor_cnt, project_cnt, diff_project, diff_donor
gc.collect()
# Merge data
df_pdd = df_donations.reset_index().merge(df_donors, on='Donor ID', how='left').set_index('Donation ID')
df_pdd = df_donations.reset_index().merge(df_projects, on='Project ID', how='left').set_index('Donation ID')
print(df_pdd.shape)

# Delete projects that are not included in projects file
df_pdd = df_pdd.loc[df_pdd['Project ID'].isin(df_projects['Project ID'].values)]
print(df_pdd.shape)
df_pdd[['Project Posted Date','Project Expiration Date','Project Fully Funded Date','Donation Received Date']].isnull().sum()
df_pdd.sort_values(['Donation Received Date', 'Project Posted Date'],inplace=True)
if debug:
    df_pdd = df_pdd.head(50000)
print(df_pdd['Project Posted Date'].min())
print(df_pdd['Project Posted Date'].max())
print(df_pdd['Donation Received Date'].min())
print(df_pdd['Donation Received Date'].max())
df_tmp = (pd.to_datetime(df_pdd['Donation Received Date']) - pd.to_datetime(df_pdd['Project Posted Date'])).dt.days
df_tmp = df_tmp.to_frame().rename(columns={0:'delay'})
df_tmp.loc[df_tmp['delay']<0, 'delay'].count()
# transform date into integer for better performance
date_feats = ['Project Posted Date','Project Expiration Date','Project Fully Funded Date','Donation Received Date']
for i in date_feats:
    df_pdd[i] = pd.to_datetime(df_pdd[i])
    df_pdd[i] = (df_pdd[i].astype(np.int64,copy=False)// 10 ** 9).astype(np.int32,copy=False)
def get_event_strength(df):
    """Calculate event strength based on Donation-Amount"""
    # Clip data
    small = 1
    large = 1000
    
    df['event_strength'] = df['Donation Amount'].clip(small, large).apply(lambda x: np.log2(1+x))
    df['event_strength'] = df['event_strength'].astype('float32', copy=False)
    return df
# Get the event strength between donor and project
df_tmp = df_pdd.groupby(['Donor ID', 'Project ID'])['Donation Amount'].sum().reset_index()
df_tmp = get_event_strength(df_tmp)
df_tmp.drop('Donation Amount', axis=1, inplace=True)
df_pde = df_tmp[['Donor ID', 'Project ID', 'event_strength']]

# df_pde['event_strength'].describe()
# sns.kdeplot(df_pde['event_strength'])
# get Donation-Received-Date
df_tmp = df_pdd.groupby(['Donor ID', 'Project ID'])['Donation Received Date'].max().reset_index()
df_pde = df_pde.merge(df_tmp, on=['Donor ID', 'Project ID'], how='left')

# Indexing by Donor Id to speed up the searches during evaluation
df_pde = df_pde.set_index('Donor ID')
df_pde.head()
df_pde.info()
del df_tmp
gc.collect()
# The commented code get a timestamp of local timezone.
# dateline = '2018-01-01'
# dateline = int(time.mktime(time.strptime(dateline, '%Y-%m-%d')))
dateline = int((datetime(2013,2,17) - datetime(1970, 1, 1)).total_seconds())
# df_pde_train, df_pde_test = train_test_split(df_pde, test_size=0.1, random_state=28)
df_pde_train = df_pde.loc[df_pde['Donation Received Date']<dateline]
df_pde_test = df_pde.loc[df_pde['Donation Received Date']>=dateline]
print('# Donations in total: %d' % len(df_pde))
print('# Donations on Train set: %d' % len(df_pde_train))
print('# Donations on Test set: %d' % len(df_pde_test))
df_pde_train.head()
#Computes the most popular items
df_project_popularity = df_pde.groupby('Project ID')['event_strength'].sum().sort_values(ascending=False).reset_index()
df_project_popularity = df_project_popularity.rename(columns={'event_strength':'rec_strength'})
df_project_popularity.head()
class PopularityRecommender(object):
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, df_popularity):
        self.df_popularity = df_popularity
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_projects(self, projects_to_ignore=[], projects_to_rec=[], topn=10):
        # We add argument 'projects_to_rec' to represent projects that can be recommended here. 
        # when it is not null, it precedes over argument 'projects_to_ignore'
        # Recommend the more popular items that the user hasn't seen yet.
        if len(projects_to_rec) == 0:
            df_recommendations = self.df_popularity[~self.df_popularity['Project ID'].isin(projects_to_ignore)] \
                                 .sort_values('rec_strength', ascending = False) \
                                 .head(topn)
        else:
            df_recommendations = self.df_popularity[self.df_popularity['Project ID'].isin(projects_to_rec)] \
                                 .sort_values('rec_strength', ascending = False) \
                                 .head(topn)

        return df_recommendations
popularity_model = PopularityRecommender(df_project_popularity)
pop_recommand = popularity_model.recommend_projects()
pop_recommand = pop_recommand.merge(df_projects, how = 'left', on = 'Project ID')\
                [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
pop_recommand
del pop_recommand
gc.collect()
text_features = ['Project Title', 'Project Essay']
df_projects['text'] = ''
for i, v in enumerate(text_features):
    #df_projects[v] = df_projects[v].astype('str', copy=False)
    df_projects[v] = df_projects[v].fillna('')
    if i < len(text_features)-1:
        df_projects['text'] +=  df_projects[v] + ' '
    else:
        df_projects['text'] += df_projects[v]
# https://drive.google.com/file/d/0B1yuv8YaUVlZZ1RzMFJmc1ZsQmM/view
# Aphost lookup dict
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}

# https://stackoverflow.com/questions/35118596/python-regular-expression-not-working-properly
pattern = r"""
           (?x)                   # set flag to allow verbose regexps
           (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
           |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
           |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
           |(?:[+/\-@&*])         # special characters with meanings        
           """
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

from nltk.tokenize.regexp import RegexpTokenizer
tokenizer=RegexpTokenizer(pattern)
def clean(content):
    """
    This function receives text contents and returns clean word-list
    """
    
    content = content.lower()                 #Convert to lower case
    content = re.sub('(\\n|\\r)','',content)  #remove \n and \r
    words = tokenizer.tokenize(content)       #Split the sentences into words
    
    # aphostophe  replacement:  you're --> you are  
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, 'v') for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_text = ' '.join(words)
    return(clean_text)
# clean data
clean_text = df_projects['text'].apply(lambda x :clean(x))

# TF-IDF
word_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=5000,
                                  lowercase=False,
                                  min_df=0.00005, max_df=0.9, 
                                  stop_words='english',
                                  strip_accents='unicode',
                                  use_idf=1, smooth_idf=1, sublinear_tf=1)
tfidf_matrix = word_vectorizer.fit_transform(clean_text)

tfidf_feature_names = word_vectorizer.get_feature_names()
project_ids = df_projects['Project ID'].tolist()
tfidf_matrix
tfidf_feature_names
del clean_text, word_vectorizer
gc.collect()
class DonorProfile(object):
    def __init__(self, tfidf_matrix, project_ids, df_pde_train):
        self.tfidf_matrix = tfidf_matrix
        self.project_ids = project_ids
        self.df_pde_train = df_pde_train
        
    def get_project_profile(self, project_id):
        idx = self.project_ids.index(project_id)
        project_profile = self.tfidf_matrix[idx:idx+1]
        return project_profile
    
    def get_project_profiles(self, ids):
        project_profiles_list = [self.get_project_profile(x) for x in np.ravel([ids])]
        project_profiles = scipy.sparse.vstack(project_profiles_list)
        return project_profiles
    
    def build_donor_profile(self, donor_id, df):
        df_tmp = df.loc[donor_id]
        donor_project_profiles = self.get_project_profiles(df_tmp['Project ID'])
        donor_project_strengths = np.array(df_tmp['event_strength']).reshape(-1,1)
        
        # Weighted average of project profiles by the donations strength
        donor_project_strengths_weighted_avg = np.sum(donor_project_profiles.multiply(donor_project_strengths), axis=0) / \
        (np.sum(donor_project_strengths)+1)
        donor_profile_norm = normalize(donor_project_strengths_weighted_avg)
        return donor_profile_norm
    
    def build_donor_profiles(self): 
        # We should use the donations in train datset to get donor profile
        # Here we can only get profiles of donors who have made donations in the train dataset
        donor_profiles = {}
        for donor_id in tqdm(self.df_pde_train.index.unique()):
            donor_profiles[donor_id] = self.build_donor_profile(donor_id, self.df_pde_train)
        return donor_profiles
donor_profile = DonorProfile(tfidf_matrix, project_ids, df_pde_train)
donor_profiles = donor_profile.build_donor_profiles()
donor_id = '0004ceb1d06fd98f0ba0364cbd7f8bdc'
relevance = donor_profiles[donor_id].flatten()
token = tfidf_feature_names

df_tmp = pd.DataFrame({'token':token, 'relevance':relevance}, columns=['token','relevance']).sort_values('relevance', ascending=False)
df_tmp.head(10)
del donor_profile, relevance, token, df_tmp
gc.collect()
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, popularity_model, tfidf_matrix, project_ids):
        self.popularity_model = popularity_model
        self.tfidf_matrix = tfidf_matrix
        self.project_ids = project_ids
        
    def get_model_name(self):
        return self.MODEL_NAME

        
    def recommend_projects(self, donor_id, projects_to_ignore=[], projects_to_rec=[], topn=10):
        if donor_id in donor_profiles:
            # Computes the cosine similarity between the donor profile and all project profiles
            cosine_similarities = cosine_similarity(donor_profiles[donor_id], self.tfidf_matrix)
            # Gets the top similar projects
            similarity = cosine_similarities.argsort().flatten().tolist()[::-1]
            ids = [self.project_ids[i] for i in similarity]
            rec_strength = cosine_similarities[0,similarity]
            df_recommendations = pd.DataFrame({'Project ID':ids, 'rec_strength':rec_strength})
            # Ignores projects in projects_to_ignore and sort the projects by similarity
            if len(projects_to_rec) == 0:
                df_recommendations = df_recommendations[~df_recommendations['Project ID'].isin(projects_to_ignore)]\
                                     .sort_values(by='rec_strength', ascending=False).head(topn)
            else:
                df_recommendations = df_recommendations[df_recommendations['Project ID'].isin(projects_to_rec)]\
                                     .sort_values(by='rec_strength', ascending=False).head(topn)
        else:
            df_recommendations = self.popularity_model.recommend_projects(projects_to_ignore, projects_to_rec, topn)

        return df_recommendations
cbr_model = ContentBasedRecommender(popularity_model, tfidf_matrix, project_ids)
cbr_recommend = cbr_model.recommend_projects(donor_id)
cbr_recommend = cbr_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
cbr_recommend
cbr_recommend = cbr_model.recommend_projects('0016b23800f7ea46424b3254f016007a')
cbr_recommend = cbr_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
cbr_recommend
del cbr_recommend
gc.collect()
# Creating a sparse pivot table with donors in rows and projects in columns
df_donors_projects_pivot = df_pde_train.reset_index().pivot(index='Donor ID', columns='Project ID', values='event_strength').fillna(0)
# Transform the donor-project dataframe into a matrix
donors_projects_pivot_matrix = df_donors_projects_pivot.values

# Get donor ids
donors_ids = df_donors_projects_pivot.index.tolist()

# Print the first 5 rows of the donor-project matrix
donors_projects_pivot_matrix[:5]
# Performs matrix factorization of the original donor-project matrix
# Here we set k = 20, which is the number of factors we are going to get
# In the definition of SVD, an original matrix A is approxmated as a product A ≈ UΣV 
# where U and V have orthonormal columns, and Σ is non-negative diagonal.
U, sigma, Vt = svds(donors_projects_pivot_matrix, k = 20)
sigma = np.diag(sigma)

# Reconstruct the matrix by multiplying its factors
all_donor_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

#Converting the reconstructed matrix back to a Pandas dataframe
df_cf_preds = pd.DataFrame(all_donor_predicted_ratings, 
                           columns = df_donors_projects_pivot.columns, 
                           index=donors_ids).transpose()
df_cf_preds.head()
del U, sigma, Vt, donors_projects_pivot_matrix, all_donor_predicted_ratings
gc.collect()
class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, popularity_model, df_cf_predictions):
        self.popularity_model = popularity_model
        self.df_cf_predictions = df_cf_predictions
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_projects(self, donor_id, projects_to_ignore=[], projects_to_rec=[], topn=10):
        if donor_id in df_cf_preds.columns:
            # Get and sort the donor's predictions
            donor_predictions = self.df_cf_predictions[donor_id].reset_index().rename(columns={donor_id: 'rec_strength'})

            if len(projects_to_rec) == 0:
                # Recommend the highest predicted projects that the donor hasn't donated to
                df_recommendations = donor_predictions[~donor_predictions['Project ID'].isin(projects_to_ignore)] \
                                     .sort_values('rec_strength', ascending = False) \
                                     .head(topn)
            else:
                df_recommendations = donor_predictions[donor_predictions['Project ID'].isin(projects_to_rec)] \
                                     .sort_values('rec_strength', ascending = False) \
                                     .head(topn)                
        else:
            df_recommendations = self.popularity_model.recommend_projects(projects_to_ignore, projects_to_rec, topn)
            
        return df_recommendations
cfr_model = CFRecommender(popularity_model, df_cf_preds)
cfr_recommend = cfr_model.recommend_projects(donor_id)
cfr_recommend = cfr_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
cfr_recommend
cfr_recommend = cfr_model.recommend_projects('0016b23800f7ea46424b3254f016007a')
cfr_recommend = cfr_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
cfr_recommend
del cfr_recommend
gc.collect()
class HybridRecommender:
    
    MODEL_NAME = 'Hybrid'
    
    def __init__(self, cbr_model, cfr_model):
        self.cbr_model = cbr_model
        self.cfr_model = cfr_model
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_projects(self, donor_id, projects_to_ignore=[], projects_to_rec=[], topn=10):
        # Getting the top-1000 Content-based filtering recommendations
        df_cbr_recs = self.cbr_model.recommend_projects(donor_id, projects_to_ignore, projects_to_rec, topn=1000)\
                      .rename(columns={'rec_strength': 'rec_strength_cb'})
        
        # Getting the top-1000 Collaborative filtering recommendations
        df_cfr_recs = self.cfr_model.recommend_projects(donor_id, projects_to_ignore, projects_to_rec, topn=1000)\
                      .rename(columns={'rec_strength': 'rec_strength_cf'})
        
        # Combining the results by Project ID
        df_recommendations = df_cbr_recs.merge(df_cfr_recs, on = 'Project ID', how = 'inner')
        
        # Computing a hybrid recommendation score based on CF and CB scores
        df_recommendations['rec_strength'] = df_recommendations['rec_strength_cb'] * df_recommendations['rec_strength_cf']
        
        # Sorting recommendations by hybrid score
        df_recommendations = df_recommendations.sort_values('rec_strength', ascending=False).head(topn)

        return df_recommendations
hybrid_model = HybridRecommender(cbr_model, cfr_model)
hybrid_recommend = hybrid_model.recommend_projects(donor_id)
hybrid_recommend = hybrid_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                   [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
hybrid_recommend
# donor '0016b23800f7ea46424b3254f016007a' can be only found in test dataset
hybrid_recommend = hybrid_model.recommend_projects('0016b23800f7ea46424b3254f016007a')
hybrid_recommend = hybrid_recommend.merge(df_projects, how = 'left', on = 'Project ID')\
                   [['rec_strength', 'Project ID', 'Project Title', 'Project Essay']]
hybrid_recommend
def get_invalid_projects(df, upper_boundary, lower_boundary=dateline):
    """Get projects that are expired or have been fully funded or haven't posted yet"""
    invalid_projects = df.loc[(df['Project Expiration Date'].notnull()) & (
                              (df['Project Expiration Date']<lower_boundary) | \
                              (df['Project Fully Funded Date']<lower_boundary) | \
                              (df['Project Posted Date']>upper_boundary)
                             )]['Project ID']
    return set(invalid_projects if type(invalid_projects) == pd.Series else [invalid_projects])
feats = ['Project Expiration Date', 'Project Fully Funded Date', 'Project Posted Date']
df_tmp = df_projects[feats+['Project ID']]
for i in feats:
    df_tmp[i] = pd.to_datetime(df_tmp[i])
    df_tmp[i] = (df_tmp[i].astype(np.int64,copy=False)// 10 ** 9).astype(np.int32,copy=False)

upper_boundary = df_pdd['Donation Received Date'].max()
invalid_projects = get_invalid_projects(df_tmp, upper_boundary)
valid_projects = set(project_ids) - invalid_projects
print(len(invalid_projects))
del df_tmp
gc.collect()
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_PROJECTS = 100

class ModelEvaluator(object):
    
    def __init__(self, model, invalid_projects, valid_projects, df_pde, df_pde_train, df_pde_test):
        self.model = model
        self.invalid_projects = invalid_projects
        self.valid_projects = valid_projects
        self.df_pde = df_pde
        self.df_pde_train = df_pde_train
        self.df_pde_test = df_pde_test
        # When set invalid_projects is larger than set valid_projects, argument 'projects_to_rec' is used.
        # or argument 'projects_to_ignore' is used.
        if len(self.invalid_projects) > len(self.valid_projects):
            self.more_set = 'invalid'
        else:
            self.more_set = 'valid'
    
    def get_interacted_projects(self, donor_id, df):
        """Get interacted projects"""
        try:
            interacted_projects = df.loc[donor_id]['Project ID'] 
            # interacted projects may be one project-id or a pandas series conposed of several project-ids
            return set(interacted_projects if type(interacted_projects) == pd.Series else [interacted_projects])
        except:
            # For a new donor, donor_id may not be included in df
            return set()
    
    def get_not_interacted_projects_sample(self, donor_id, sample_size, seed=42):
        interacted_projects = self.get_interacted_projects(donor_id, self.df_pde)
        non_interacted_projects = self.valid_projects - interacted_projects
        
        random.seed(seed)
        non_interacted_projects_sample = random.sample(non_interacted_projects, sample_size)
        return set(non_interacted_projects_sample)

    def _verify_topn_hit(self, item_id, recommended_projects, topn):        
        try:
            index = next(i for i, c in enumerate(recommended_projects) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index
        """
        index = -1
        for i, c in enumerate(recommended_projects): 
            if c == 3:
                index = i
                break
        """

    def evaluate_model_for_donor(self, donor_id):
        # Getting the projects in test set
        donor_interacted_test = self.df_pde_test.loc[donor_id]
        if type(donor_interacted_test['Project ID']) == pd.Series:
            donor_interacted_projects_test = set(donor_interacted_test['Project ID'])
        else:
            donor_interacted_projects_test = set([donor_interacted_test['Project ID']])  
        interacted_projects_test_count = len(donor_interacted_projects_test) 

        # Getting a ranked recommendation list from a model for a given donor
        if self.more_set == 'valid':
            projects_to_ignore = self.invalid_projects.union(self.get_interacted_projects(donor_id, self.df_pde_train))
            df_recs = self.model.recommend_projects(donor_id, projects_to_ignore, [], topn=10000000000)
        else:
            projects_to_rec = self.valid_projects - self.get_interacted_projects(donor_id, self.df_pde_train)
            df_recs = self.model.recommend_projects(donor_id, [], projects_to_rec, topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        # For each project the donor has interacted in test set
        for item_id in donor_interacted_projects_test:
            # Getting a random sample (100) projects the donor has not interacted 
            # (to represent projects that are assumed to be no relevant to the user)
            non_interacted_projects_sample = self.get_not_interacted_projects_sample(donor_id, 
                                                                                     sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_PROJECTS)

            # Combining the current interacted project with the 100 random projects
            projects_to_filter_recs = non_interacted_projects_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted project or from a random sample of 100 non-interacted projects
            recs_valid = df_recs.loc[df_recs['Project ID'].isin(projects_to_filter_recs), 'Project ID'].values
            #Verifying if the current interacted project is among the Top-N recommended projects
            hit_at_5, index_at_5 = self._verify_topn_hit(item_id, recs_valid, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_topn_hit(item_id, recs_valid, 10)
            hits_at_10_count += hit_at_10

        # Recall is the rate of the interacted projects that are ranked among the Top-N recommended projects, 
        # when mixed with a set of non-relevant projects
        recall_at_5 = hits_at_5_count / float(interacted_projects_test_count)
        recall_at_10 = hits_at_10_count / float(interacted_projects_test_count)

        donor_metrics = {'_donor_id':donor_id,
                         'hits@5_count':hits_at_5_count, 
                         'hits@10_count':hits_at_10_count, 
                         'interacted_count': interacted_projects_test_count,
                         'recall@5': recall_at_5,
                         'recall@10': recall_at_10}
        return donor_metrics


    def evaluate_model(self):
        #print('Running evaluation for donors')
        donors_metrics = []
        ids = self.df_pde_test.index.unique().tolist()
        print('%d donors to be processed' %len(ids))
        for idx, donor_id in enumerate(ids):
            if idx % 100 == 0 and idx > 0:
                print('%d donors processed' % idx)
            donor_metrics = self.evaluate_model_for_donor(donor_id)
            donors_metrics.append(donor_metrics)
        
        """
        ids = self.df_pde_test.index.unique().tolist()
        pool = Pool(4)
        donors_metrics = pool.map(self.evaluate_model_for_donor, ids)
        pool.close()
        pool.join()
        """

        df_results = pd.DataFrame(donors_metrics).sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = df_results['hits@5_count'].sum() / float(df_results['interacted_count'].sum())
        global_recall_at_10 = df_results['hits@10_count'].sum() / float(df_results['interacted_count'].sum())
        
        global_metrics = {'modelName': self.model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, df_results
  
model_evaluator = ModelEvaluator(cbr_model, invalid_projects, valid_projects, df_pde, df_pde_train, df_pde_test)

start = datetime.now()
print(start)

cbr_global_metrics, cbr_df_results = model_evaluator.evaluate_model()
for k, v in cbr_global_metrics.items():
    print('%s: %s' %(k,v))

end = datetime.now()
print(end)

cbr_df_results.head(10)
model_evaluator = ModelEvaluator(cfr_model, invalid_projects, valid_projects, df_pde, df_pde_train, df_pde_test)

start = datetime.now()
print(start)

cfr_global_metrics, cfr_df_results = model_evaluator.evaluate_model()
for k, v in cfr_global_metrics.items():
    print('%s: %s' %(k,v))

end = datetime.now()
print(end)

cfr_df_results.head(10)
model_evaluator = ModelEvaluator(hybrid_model, invalid_projects, valid_projects, df_pde, df_pde_train, df_pde_test)

start = datetime.now()
print(start)

hybrid_global_metrics, hybrid_df_results = model_evaluator.evaluate_model()
for k, v in hybrid_global_metrics.items():
    print('%s: %s' %(k,v))

end = datetime.now()
print(end)

hybrid_df_results.head(10)
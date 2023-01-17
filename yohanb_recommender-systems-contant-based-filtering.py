import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
articles_df = pd.read_csv('../input/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(2)
interactions_df = pd.read_csv('../input/users_interactions.csv')
interactions_df.head(2)
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])
users_interactions_count_df = interactions_df.groupby(['personId', 'contentId'])\
                                             .size()\
                                             .groupby('personId')\
                                             .size()
print('Total # of users that we get: {}'.format(len(users_interactions_count_df)))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5]\
                                             .reset_index()[['personId']]
print('Users with >= 5 interactions: {}'.format(len(users_with_enough_interactions_df)))
print('Total # of interactions: {}'.format(len(interactions_df)))
interactions_from_selected_users_df = interactions_df\
                                        .merge(users_with_enough_interactions_df, 
                                               how = 'right',
                                               left_on = 'personId',
                                               right_on = 'personId')
print('Interactions from users with >= 5: {}'.format(len(interactions_from_selected_users_df)))
def smooth_user_preference(x):
    return math.log(1+x,2)
    
interactions_full_df = interactions_from_selected_users_df \
                        .groupby(['personId', 'contentId', 'timestamp'])['eventStrength']\
                        .sum()\
                        .apply(smooth_user_preference)\
                        .reset_index()
interactions_full_df.head(5)
# Interaction before 2016-09-01:
interactions_train_df = interactions_full_df[pd.to_datetime(
                                                interactions_full_df.timestamp, 
                                                unit='s')\
                                             < pd.Timestamp(2016,9,1)].copy()
# Interaction after 2016-09-01:
interactions_test_df = interactions_full_df[(pd.to_datetime(
                                                interactions_full_df.timestamp,
                                                unit='s')\
                                             >= pd.Timestamp(2016,9,1))].copy()

# Here we also filter out all users from test set that are not in the train set: 
interactions_test_df = interactions_test_df[np.isin(interactions_test_df.personId, interactions_train_df.personId)]
interactions_train_df = interactions_train_df[np.isin(interactions_train_df.personId, interactions_test_df.personId)]
train_len = len(interactions_train_df)
test_len = len(interactions_test_df)
full_len = train_len + test_len

print('# interactions on Train set: {:.2f}% [{} in total]'.format(train_len/full_len*100, train_len))
print('# interactions on Test  set: {:.2f}% [{} in total]'.format(test_len/full_len*100, test_len))
#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')
def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual.all():
        return 0.0
    return score / min(len(actual), k)
#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)
        
        #Additional: map@5 and map@10
        ap_5 = apk(np.append([], interacted_values_testset.contentId.tolist()), person_recs_df.contentId.tolist()[:5], 5)
        ap_10 = apk(np.append([], interacted_values_testset.contentId.tolist()), person_recs_df.contentId.tolist()[:10], 10)
        
        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10,
                          'ap@5': ap_5,
                          'ap@10': ap_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            if idx % 100 == 0 and idx > 0:
                print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        #map@k
        map_at_5 = detailed_results_df['ap@5'].values.mean()
        map_at_10 = detailed_results_df['ap@10'].values.mean()
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10,
                          'map@5': map_at_5,
                          'map@10': map_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()    
#Ignoring stopwords (words with no semantics) from English and Portuguese (as we have a corpus with mixed languages)
stopwords_list = stopwords.words('english') + stopwords.words('portuguese')

#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.003,
                     max_df=0.5,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])
    
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'] \
                                                   .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles
user_profiles = build_users_profiles()
len(user_profiles)
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(articles_df)
print('Evaluating Content-Based Filtering model...')
cb_global_metrics_1, cb_detailed_results_df_1 = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics_1)
cb_detailed_results_df_1.sample(10)
from sklearn.preprocessing import MinMaxScaler
minmaxScaler = MinMaxScaler()

# Get length of the title and text
articles_df['title_len'] = articles_df.title.map(len)
articles_df['text_len'] = articles_df.text.map(len)
# We use the min-amx scaler here because TF-IDF data is sparse and every value there is less than 0.5
articles_df_dummies_titletext_len = minmaxScaler.fit_transform(articles_df[['title_len', 'text_len']])
text_art_names = ['title_len', 'text_len']

# Get infornation about authors' regions
articles_df_dummies_region = pd.get_dummies(
                                articles_df.authorRegion.map(
                                        lambda x: np.nan
                                        if x not in frozenset(['SP', 'MG', 'NY', 'NJ'])
                                        else x))
region_names = articles_df_dummies_region.columns.tolist()

# Get infornation about authors' country
articles_df_dummies_country = pd.get_dummies(articles_df.authorCountry)
country_names = articles_df_dummies_country.columns.tolist()

# Get the list of authors (we consider only authors that have at least 20 articles)
authors_num_art = articles_df.authorPersonId.value_counts()
articles_df_dummies_author = pd.get_dummies(
                                articles_df.authorPersonId.map(
                                        lambda x: np.nan 
                                        if authors_num_art[x] < 20 
                                        else 'author_' + str(x)))
authors_names = ['A'+str(i) for i in range(articles_df_dummies_author.shape[1])]

# Get infornation about the day of week
day_of_week = pd.get_dummies(pd.to_datetime(articles_df.timestamp, unit='s')\
                             .dt.weekday.map(
                                {0:'Mo',1:'Tu',2:'Wd',3:'Th',4:'Fr',5:'Su',6:'Sa'}))
day_of_week_names = day_of_week.columns.tolist()

# Concatinate all the new information into one table
articles_df_dummies_all = np.hstack([articles_df_dummies_titletext_len, 
                                     articles_df_dummies_region, 
                                     articles_df_dummies_country, 
                                     articles_df_dummies_author,
                                     day_of_week])
names_all = text_art_names + region_names + country_names \
            + authors_names + day_of_week_names
print(names_all, len(names_all))
# Augmintation
from scipy.sparse import csr_matrix
tfidf_matrix = csr_matrix(scipy.sparse.hstack([0.3 * articles_df_dummies_all, tfidf_matrix]))
tfidf_matrix
user_profiles = build_users_profiles()
len(user_profiles)
print('Evaluating Content-Based Filtering model...')
cb_global_metrics_2, cb_detailed_results_df_2 = model_evaluator.evaluate_model(content_based_recommender_model)
print('\nGlobal metrics:\n%s' % cb_global_metrics_2)
cb_detailed_results_df_2.sample(10)
fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.suptitle('Recall@5')
fig.set_figwidth(15)
# 1
axarr[0].hist(cb_detailed_results_df_1['recall@5'], rwidth=0.9);
axarr[0].set_title('Raw TF-IDF');
# 2
axarr[1].hist(cb_detailed_results_df_2['recall@5'], rwidth=0.9, color='green');
axarr[1].set_title('Augmented TF-IDF');
# 3
axarr[2].hist(cb_detailed_results_df_1['recall@5'], rwidth=0.9, color='blue', alpha=0.5);
axarr[2].hist(cb_detailed_results_df_2['recall@5'], rwidth=0.9, color='green', alpha=0.5);
axarr[2].set_title('Raw vs. Augmented');
fig, axarr = plt.subplots(nrows=1, ncols=3, sharey=True)
fig.suptitle('AP@5')
fig.set_figwidth(15)
# 1
axarr[0].hist(cb_detailed_results_df_1['ap@5'], rwidth=0.9);
axarr[0].set_title('Raw TF-IDF');
# 2
axarr[1].hist(cb_detailed_results_df_2['ap@5'], rwidth=0.9, color='green');
axarr[1].set_title('Augmented TF-IDF');
# 3
axarr[2].hist(cb_detailed_results_df_1['ap@5'], rwidth=0.9, color='blue', alpha=0.5);
axarr[2].hist(cb_detailed_results_df_2['ap@5'], rwidth=0.9, color='green', alpha=0.5);
axarr[2].set_title('Raw vs. Augmented');
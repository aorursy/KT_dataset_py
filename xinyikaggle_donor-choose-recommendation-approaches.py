# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import re
import collections
import string
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
# Load data
projects = pd.read_csv('../input/Projects.csv')
donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv')
# Drop duplicates 
projects = projects.drop_duplicates('Project ID')
# Sample data for test
SEED = 42
random.seed(SEED)
test_mode = True
sample_idx = random.sample(projects.index.tolist(), math.floor(projects.shape[0] * 0.01))
projects_sample = projects.iloc[sample_idx]
projects_sample.index = range(projects_sample.shape[0])
sample_idx_donor = random.sample(donors.index.tolist(), math.floor(donors.shape[0] * 0.01))
donors_sample = donors.iloc[sample_idx_donor]
donors_sample.index = range(donors_sample.shape[0])

if not test_mode:
    projects_sample = projects
    donors_sample = donors

donations_donors = pd.merge(donations, donors_sample,
    how='inner', on='Donor ID')
donations_donors_projects = pd.merge(donations_donors, projects_sample, 
    how='inner', on='Project ID')
donations_donors_projects = donations_donors_projects[[
    'Donor ID', 'Project ID', 'Donation ID', 'Donation Amount']]
donations_donors_projects = donations_donors_projects.groupby([
    'Donor ID', 'Project ID'], as_index=False)['Donation Amount'].sum()
donations_donors_projects['Donation Amount'] = donations_donors_projects[
    'Donation Amount'].apply(lambda x: math.log(1+x))
donations_donors_projects.index = range(donations_donors_projects.shape[0])
donations_donors_projects.head(3)
project_ids = projects_sample['Project ID'].tolist()

print("Project numbers: {}".format(len(project_ids)))
donor_ids = donations_donors_projects['Donor ID'].unique().tolist()
# Split train test data
train_data, test_data = train_test_split(donations_donors_projects, test_size=0.3, random_state=42)
projects_in_train = train_data['Project ID'].tolist()
projects_sample['Text'] = projects_sample[
            'Project Title'] + projects_sample[
            'Project Essay'] + projects_sample[
            'Project Need Statement']
projects_sample['Text'] = projects_sample['Text'].astype(str)

results = []
for sentence in projects_sample['Text']:
    res = "".join(sentence.lower().strip())
    res = res.replace('<!--DONOTREMOVEESSAYDIVIDER-->', '')
    res = res.replace('need ', '')#.split(". ")[0].split("!")[0].strip().split()[:3])
    res = re.sub(r'\d+', '', res)
    results.append(res)       
projects_sample['Text'] = results
# TF-IDF Vectorization (relevence calculation)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tf.fit_transform(projects_sample['Text'])
tfidf_feature_names = tf.get_feature_names()
def get_donor_relevence(project_id, project_ids):
    project_idx = project_ids.index(project_id)
    project_relevence = tfidf_matrix[project_idx:(project_idx + 1)]
    return project_relevence

def get_weighted_donor_relevence(donor_id, donations_donors_projects):
    donor_df = donations_donors_projects[donations_donors_projects['Donor ID'] == donor_id]
    donor_df.index = range(donor_df.shape[0])
    project_indices = donor_df['Project ID'].tolist()
    donation_weights = donor_df['Donation Amount'] / sum(donor_df['Donation Amount'])  

    for i, project_id in enumerate(project_indices):
        if i < 1:
            res = get_donor_relevence(project_id, project_ids) * donation_weights[i]
        else:
            res += get_donor_relevence(project_id, project_ids) * donation_weights[i]
        
    res = preprocessing.normalize(res)
    return res

def donor_relevences(donations_donors_projects, donor_ids):
    donor_relevences_dict = {}
    for donor_id in donor_ids:
        donor_relevences_dict[donor_id] = get_weighted_donor_relevence(donor_id, donations_donors_projects)
    return donor_relevences_dict
donor_relevances_dict = donor_relevences(donations_donors_projects, donor_ids)
donor1 = donor_ids[9]
print("Example donor: {}".format(donor1))
donor1_df = pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        donor_relevances_dict[donor1].todense().tolist()[0]), 
                        key=lambda x: -x[1])[:10],
                        columns=['token', 'relevance'])
donor1_df
donor1_projects = set(donations_donors_projects[
    donations_donors_projects['Donor ID'] == donor1][
    'Project ID'])
donor1_project_ids = [project_ids.index(i) for i in donor1_projects]

donor1_projects_df = projects_sample.loc[donor1_project_ids]
donor1_projects_df
# Recommendation engine
# Include donated projects for model evaluation purposes
class ContentBasedRSEval:
    MODEL_NAME = "Content-Based"
    
    def __init__(self, project_ids, projects_df):
        self.project_ids = project_ids
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_donor_projects(self, donor_id):
        donor_project_ids = set(donations_donors_projects[
            donations_donors_projects['Donor ID'] == donor_id][
            'Project ID'])
        donor_project_ids = list(donor_project_ids)
        return donor_project_ids
    
    def _get_similar_projects(self, donor_id):
        # Calculate cosine similarity
        donor_project_ids = self._get_donor_projects(donor_id)
        cosine_similarities = cosine_similarity(donor_relevances_dict[donor_id], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort().flatten()[:-(1000 + 1):-1].tolist()
        similar_items = [(self.project_ids[i], cosine_similarities[i]) for i in similar_indices]
        return similar_items
    
    def recommended_projects(self, donor_id, projects_in_train=[], recommend_num=10):
        similar_items = self._get_similar_projects(donor_id)
        recommendation_df = pd.DataFrame(similar_items, columns=['Project ID', 'Relevence'])
        recommendation_df = recommendation_df[~recommendation_df['Project ID'].isin(
            projects_in_train)].sort_values(by='Relevence', ascending=False).head(recommend_num)
        recommendation_df = pd.merge(recommendation_df, self.projects_df, how='inner', on='Project ID')
        
        return recommendation_df
rps_content_eval = ContentBasedRSEval(project_ids, projects_sample[['Project ID', 'Project Title']])
rps_content_eval.recommended_projects(donor1, projects_in_train, 15)
# Exclude donated projects.
class ContentBasedRS:
    MODEL_NAME = "Content-based"
    
    def __init__(self, project_ids, projects_df):
        self.project_ids = project_ids
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_donor_projects(self, donor_id):
        donor_project_ids = set(donations_donors_projects[
            donations_donors_projects['Donor ID'] == donor_id][
            'Project ID'])
        donor_project_ids = list(donor_project_ids)
        
        return donor_project_ids
    
    def _get_similar_projects(self, donor_id, recommend_num=1000):
        # Calculate cosine similarity
        donor_project_ids = self._get_donor_projects(donor_id)
        cosine_similarities = cosine_similarity(donor_relevances_dict[donor_id], tfidf_matrix).flatten()
        similar_indices = cosine_similarities.argsort().flatten()[:-(1000 + 1):-1].tolist()
        similar_items = [(self.project_ids[i], cosine_similarities[i]) for i in similar_indices 
                         if self.project_ids[i] not in donor_project_ids]
        return similar_items
    
    def recommended_projects(self, donor_id, recommend_num=10):
        similar_items = self._get_similar_projects(donor_id)
        recommendation_df = pd.DataFrame(similar_items, columns=['Project ID', 'Relevence']).head(recommend_num)
        recommendation_df = pd.merge(recommendation_df, self.projects_df, how='inner', on='Project ID')
        
        return recommendation_df
rps_content = ContentBasedRS(project_ids, projects_sample[['Project ID', 'Project Title']])
rps_content.recommended_projects(donor1, 15)
pivot_table = pd.pivot_table(donations_donors_projects, 
                values='Donation Amount', index='Donor ID', 
                columns='Project ID').fillna(0)#, aggfunc=np.sum)
pivot_table_matrix = pivot_table.as_matrix()

u, s, vt = svds(pivot_table_matrix, k=20)
s_diag_matrix=np.diag(s)
pivot_predicted = np.dot(np.dot(u, s_diag_matrix), vt)

pivot_predicted_df = pd.DataFrame(pivot_predicted, 
                           columns = pivot_table.columns, 
                           index=donor_ids).transpose()
pivot_predicted_df.reset_index(level=0, inplace=True)
class SVDRSEVAL:
    MODEL_NAME = "SVD"
    
    def __init__(self, project_ids, projects_df):
        self.pivot_predicted = pivot_predicted_df
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def _get_donor_projects(self, donor_id):
        donor_project_ids = set(donations_donors_projects[
            donations_donors_projects['Donor ID'] == donor_id][
            'Project ID'])
        donor_project_ids = list(donor_project_ids)
        return donor_project_ids
    
    def recommended_projects(self, donor_id, projects_in_train=[], recommend_num=1000):
        recommended_projects = pivot_predicted_df[['Project ID', donor_id]].sort_values(by = donor_id,
            ascending=False).head(1000)
        
        recommended_projects.columns = ['Project ID','Relevence']
        recommendation_df = recommended_projects[~recommended_projects['Project ID'].isin(
            projects_in_train)].sort_values(by='Relevence', ascending=False).head(recommend_num)
        recommendation_df = pd.merge(recommendation_df, self.projects_df, how='inner', on='Project ID')
        
        return recommendation_df
rps_svd_eval = SVDRSEVAL(project_ids, projects[['Project ID', 'Project Title']])
rps_svd_eval.recommended_projects(donor1, projects_in_train, recommend_num=15)
class SVDRS:
    MODEL_NAME = "SVD"
    
    def __init__(self, project_ids, projects_df):
        self.project_ids = project_ids
        self.projects_df = projects_df
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def _get_donor_projects(self, donor_id):
        donor_project_ids = set(donations_donors_projects[
            donations_donors_projects['Donor ID'] == donor_id][
            'Project ID'])
        donor_project_ids = list(donor_project_ids)
        return donor_project_ids
    
    def recommended_projects(self, donor_id, recommend_num=1000):
        donor_project_ids = self._get_donor_projects(donor_id)
        projects_to_exlude = donor_project_ids + projects_in_train
        recommended_projects = pivot_predicted_df[['Project ID', donor_id]].sort_values(by = donor_id,
            ascending=False).head(1000)
        recommended_projects.columns = ['Project ID','Relevence']
        recommendation_df = recommended_projects[~recommended_projects['Project ID'].isin(
            projects_to_exlude)].sort_values(by='Relevence', ascending=False).head(recommend_num)
        
        recommendation_df = pd.merge(recommendation_df, self.projects_df, how='inner', on='Project ID')
        
        return recommendation_df
rps_svd = SVDRS(project_ids, projects_sample[['Project ID', 'Project Title']])
rps_svd.recommended_projects(donor1, 15)
class MixHybridEVAL:
    MODEL_NAME = "Hybrid"
    
    def __init__(self, rps_content, rps_svd):
        self.cb_approach = rps_content
        self.svd_approach = rps_svd
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _min_max_scaler(self, arr):
        scaled_arr = [0] * len(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        for i in range(len(arr)):
            scaled_arr[i] = (arr[i] - min_val) / (max_val - min_val)
        return scaled_arr
        
    def recommended_projects(self, donor_id, projects_in_train=[], recommend_num=10):
        # Content-based approach
        cb_df = self.cb_approach.recommended_projects(donor_id, projects_in_train, 1000)
        cb_df.columns = ['Project ID', 'Relevence_CB', 'Project Title_CB']
        cb_df['Relevence_CB'] = self._min_max_scaler(cb_df['Relevence_CB'].tolist())
        # SVD approach
        svd_df = self.svd_approach.recommended_projects(donor_id, projects_in_train, 1000)
        svd_df.columns = ['Project ID', 'Relevence_SVD', 'Project Title_SVD']
        svd_df['Relevence_SVD'] = self._min_max_scaler(svd_df['Relevence_SVD'].tolist())
        hybrid_df = pd.merge(cb_df, svd_df, how = 'inner', 
                                   left_on = 'Project ID', 
                                   right_on = 'Project ID')
        
        hybrid_df['Total relevence'] = 0.5 * hybrid_df['Relevence_CB'] + 0.5 * hybrid_df['Relevence_SVD']
    
        # rank again by toal relevence
        hybrid_df = hybrid_df.sort_values('Total relevence',
                        ascending=False).head(recommend_num)
        hybrid_df.drop('Project Title_SVD', axis=1, inplace=True)
        hybrid_df.index = range(hybrid_df.shape[0])
        
        return hybrid_df
rps_hybrid_eval = MixHybridEVAL(rps_content_eval, rps_svd_eval)
rps_hybrid_eval.recommended_projects(donor1, projects_in_train, 15)
class MixHybrid:
    MODEL_NAME = "Hybrid"
    
    def __init__(self, rps_content, rps_svd):
        self.cb_approach = rps_content
        self.svd_approach = rps_svd
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def _min_max_scaler(self, arr):
        scaled_arr = [0] * len(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        for i in range(len(arr)):
            scaled_arr[i] = (arr[i] - min_val) / (max_val - min_val)
        return scaled_arr
        
    def recommended_projects(self, donor_id, recommend_num=10):
        # Content-based approach
        cb_df = self.cb_approach.recommended_projects(donor_id, 1000)
        cb_df.columns = ['Project ID', 'Relevence_CB', 'Project Title_CB']
        cb_df['Relevence_CB'] = self._min_max_scaler(cb_df['Relevence_CB'].tolist())
        # SVD approach
        svd_df = self.svd_approach.recommended_projects(donor_id, 1000)
        svd_df.columns = ['Project ID', 'Relevence_SVD', 'Project Title_SVD']
        svd_df['Relevence_SVD'] = self._min_max_scaler(svd_df['Relevence_SVD'].tolist())
        hybrid_df = pd.merge(cb_df, svd_df, how = 'inner', 
                                   left_on = 'Project ID', 
                                   right_on = 'Project ID')
        
        hybrid_df['Total relevence'] = 0.5 * hybrid_df['Relevence_CB'] + 0.5 * hybrid_df['Relevence_SVD']
    
        # rank again by toal relevence
        hybrid_df = hybrid_df.sort_values('Total relevence',
                        ascending=False).head(recommend_num)
        hybrid_df.drop('Project Title_SVD', axis=1, inplace=True)
        
        return hybrid_df
rps_hybrid = MixHybrid(rps_content, rps_svd)
rps_hybrid.recommended_projects(donor1, 15)
# Top-N Accuracy metric functions
SAMPLE_SIZE = 100
def _get_dononated_projects(donor_id):
    donor_project_ids = test_data[
        test_data['Donor ID'] == donor_id][
        'Project ID'].tolist()
    return donor_project_ids
    
def _get_other_projects(donor_id):
    donated_project_ids = _get_dononated_projects(donor_id)
    other_project_ids = [i for i in project_ids if i not in donated_project_ids]
    other_project_ids = random.sample(other_project_ids, SAMPLE_SIZE)
    return other_project_ids
    
def _hit_top_n(project_id, recommended_projects, topn):
    index = -1
    for i, project in enumerate(recommended_projects):
        if project == project_id:
            index = i
            break
    return int(index in range(topn))
    
def _hit_count_for_donor(model, donor_id):
    donated_proejcts = _get_dononated_projects(donor_id)
    donated_count = len(donated_proejcts)
    recommended_df = model.recommended_projects(donor_id, 
                            projects_in_train, 
                            recommend_num = 100)
        #recomd_projects = recommended_df['Project ID'].tolist()
        
    hit_at_1_count, hit_at_5_count, hit_at_10_count = 0, 0, 0
    for project_id in donated_proejcts:
        projects_not_donated = _get_other_projects(donor_id)
        projects_to_test = projects_not_donated + [project_id]
        recommeded_in_test_df = recommended_df[recommended_df['Project ID'].isin(projects_to_test)]
        recommeded_in_test_projects = recommeded_in_test_df['Project ID'].tolist()
        
        #print("project id", project_id)
        #print(recommeded_in_test_projects)
            
        hit_at_1_count += _hit_top_n(project_id, recommeded_in_test_projects, 1)
        hit_at_5_count += _hit_top_n(project_id, recommeded_in_test_projects, 5)
        hit_at_10_count += _hit_top_n(project_id, recommeded_in_test_projects, 10)
        
        #recall_at_3 = hit_at_3_count / donated_count
        #recall_at_5 = hit_at_5_count / donated_count
        #recall_at_10 = hit_at_10_count / donated_count
    hit_count_res = [hit_at_1_count, hit_at_5_count, hit_at_10_count, donated_count]    
    return hit_count_res
    
def recall_for_donors(model):
    recall_list = []
    for donor in test_data['Donor ID'].unique().tolist():
        hit_res = _hit_count_for_donor(model, donor)
        recall_list.append(hit_res)
            
    recall_df = pd.DataFrame(recall_list, columns = ['Hit_at_1','Hit_at_5', 'Hit_at_10', 'count'])
    total_count = recall_df['count'].sum()
    recall_1 = recall_df['Hit_at_1'].sum() / total_count
    recall_5 = recall_df['Hit_at_5'].sum() / total_count
    recall_10 = recall_df['Hit_at_10'].sum() / total_count
        
    recalls_res = [recall_1, recall_5, recall_10]
    recalls_df = pd.DataFrame(recalls_res, columns = ['Recall metric'])
    recalls_df.index = ['Recall_at_1', 'Recall_at_5', 'Recall_at_10']
    
    return recalls_df
recall_content = recall_for_donors(rps_content_eval)
recall_content
recall_svd = recall_for_donors(rps_svd_eval)
recall_svd
recall_hybrid = recall_for_donors(rps_hybrid_eval)
recall_hybrid
recalls_df = pd.concat([
    recall_content, recall_svd, recall_hybrid], 
    axis=1)
recalls_df.columns = ['Recall_Content-Based', 'Recall_SVD', 'Recall_Hybrid']
recalls_df
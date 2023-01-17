import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_click = pd.read_csv('/kaggle/input/jobs-recommendation-dataset/job_clicks.csv', dtype=str)

df_jobs = pd.read_csv('/kaggle/input/jobs-recommendation-dataset/jobs.csv', dtype=str)

df_click.shape
df_jobs.head(2)
# preprocess

def get_train_data(df):

    """

    :param

      df: pd.DataFrame(columns=['userId', 'jobId', 'Clicks'])

    :return

      texts: [['jobid1', 'jobid2'], ['jobid3', 'jobid9'],...]

      user_items_dict: {'userid1: ['jobid1', jobid3'], ...]

    """

    texts = [] # list for input

    user_jobs_dict = {} # dict for {user: [jobs]}



    # sort by clicks in descending order.

    for i, v in df.groupby('userId'):

        jobs = v.sort_values(by='Clicks', ascending=False)['jobId'].tolist()

        user_jobs_dict[i] = jobs

        texts.append(jobs)



    return texts, user_jobs_dict
texts, user_jobs_dict = get_train_data(df_click)
import gensim

# train item2vec

model = gensim.models.word2vec.Word2Vec(

    texts, size=50, window=200, iter=20, workers=10, min_count=1)

model.save('./w2v.model')
def rec_all_topn(model, user_jobs: dict, topn=10, syn_vec_num=3):

    def sim_list(model, user_id, job_ids):

        try:

            results = model.wv.most_similar(

                positive=job_ids[:syn_vec_num],

                topn=topn

            )

        except Exception as e:

            print(f'{job_ids} not in vocab')

            print(e)

            return []

        

        recs = []

        for result in results:

            job_id, sim = result

            rec = [user_id, job_id, sim]

            recs.append(rec)

            

        return recs



    all_recs = []

    for user_id, job_ids in user_jobs.items():

        all_recs.extend(

            sim_list(

                model=model,

                user_id=user_id,

                job_ids=job_ids

            )

        )

    df = pd.DataFrame(all_recs, columns=['user_id', 'job_id', 'sim'])

    

    return df
len(df_click['userId'].value_counts())
df_rec_top100 = rec_all_topn(model, user_jobs_dict, topn=100)

#df_rec_top10 = rec_all_topn(model, user_jobs_dict, topn=10)
df_rec_top100.shape
import collections



def mmr_sorted(docs, q, similarity1, similarity2, lam=0.5):

    """Sort a list of docs by Maximal marginal relevance



    :param docs: a set of documents to be ranked by maximal marginal relevance

    :param q: query to which the documents are results

    :param lam : lambda parameter used in computation of MMR score.

    :param similarity1: sim_1 function. takes a doc and the query

                        as an argument and computes their similarity

    :param similarity2: sim_2 function. takes two docs as arguments

                        and computes their similarity score

    :return: a (document, mmr score) ordered dictionary of the docs

             given in the first argument, ordered my MMR

    """

    selected = collections.OrderedDict()

    while set(selected) != docs:

        remaining = docs - set(selected)

        mmr_score = lambda x: lam * similarity1([x], q) - (1 - lam) * max(

            [similarity2(x, y) for y in set(selected) - {x}] or [0])

        next_selected = max(remaining, key=mmr_score)

        selected[next_selected] = len(selected)

    return selected



def w2v_n_sim(model, wordset1: list, wordset2: list):

    return model.n_similarity(wordset1, wordset2)
recs_job_ids = df_rec_top100.groupby('user_id').agg({'job_id':lambda x: set(x)})['job_id'].to_list()



mmr_users = []

for (user_id, job_ids), rec_job_ids in zip(user_jobs_dict.items(), recs_job_ids):

    mmr = [mmr_sorted(rec_job_ids, job_ids[:3], model.wv.n_similarity, model.wv.similarity)]

                                        

    mmr_users.append(mmr)
df_mmrs = pd.melt(pd.DataFrame([mmr_users[i][0] for i, j in enumerate(mmr_users)]).T.reset_index(),

                  id_vars='index').rename(columns = {'index':'job_id', 'variable': 'user_id', 'value': 'order'})

df_mmrs.shape
# recommend top100 for user_id=1 by mmr score

df_mmrs[(df_mmrs['order'].notna()) & (df_mmrs['user_id'] == 1)].sort_values(by='order')
from sklearn.metrics.pairwise import cosine_similarity

import scipy.sparse

def ils(predicted, df_feat):

    recs_content = df_feat.loc[predicted]

    recs_content = recs_content.dropna()

    recs_content = scipy.sparse.csr_matrix(recs_content.values)



    #calculate similarity scores for all items in list

    similarity = cosine_similarity(X=recs_content, dense_output=False)



    #get indicies for upper right triangle w/o diagonal

    upper_right = np.triu_indices(similarity.shape[0], k=1)



    #calculate average similarity score of all recommended items in list

    ils_single_user = np.mean(similarity[upper_right])

    return ils_single_user
# job category

df_feat_jobs = pd.get_dummies(df_jobs.set_index('jobID'))
rec_num = 30
# ILS sort by similarity

recs_job_ids = df_rec_top100.groupby('user_id').agg({'job_id':lambda x: list(x)[:rec_num]})['job_id'].to_list()



ils_users = [ils(rec_job_ids, df_feat_jobs) for rec_job_ids in recs_job_ids]

np.mean(ils_users)
# ILS sort by MMR

recs_mmr_job_ids = df_mmrs[df_mmrs['order'].notna()].groupby('user_id').agg({'job_id':lambda x: list(x)[:rec_num]})['job_id'].to_list()



ils_mmr_users = [ils(rec_job_ids, df_feat_jobs) for rec_job_ids in recs_mmr_job_ids]

np.mean(ils_mmr_users)
print(f'''intra list similarity

sim: {np.mean(ils_users):.4f}

MMR: {np.mean(ils_mmr_users):.4f}

sorted by {"MMR" if np.mean(ils_mmr_users) < np.mean(ils_users) else "sim"} is more diverse.

''')
bins = np.linspace(0, 0.6, 20)



plt.hist(ils_users, bins, alpha = 0.5, label='sort by sim')

plt.hist(ils_mmr_users, bins, alpha = 0.5, label='sort by MMR')

plt.legend()

plt.title('intra list similarity')

plt.show()
# ex.) check for user_id = '1'

user_id = '1'

q_ids = user_jobs_dict['1'][:3] # history

job_ids = set(df_rec_top100[df_rec_top100['user_id']=='1']['job_id'].to_list())

mmr_user = mmr_sorted(job_ids, q_ids, model.wv.n_similarity, model.wv.similarity)



df_mmr = pd.DataFrame(mmr_user, index=['order']).T.reset_index().rename(columns = {'index':'job_id'})

#df_mmr.head()



predicted_sim = list(df_rec_top100[df_rec_top100['user_id']=='1']['job_id'])[:30]

predicted_mmr = list(df_mmr['job_id'])[:30]



ils_user_sim = ils(predicted_sim, df_feat_jobs)

ils_user_mmr = ils(predicted_mmr, df_feat_jobs)



print(f'''ILS for user_id={user_id}, candidate jobs=100, recommendation top30 jobs.

sim: {ils_user_sim:.4f}

MMR: {ils_user_mmr:.4f}

sorted by {"MMR" if ils_user_mmr < ils_user_sim else "sim"} is more diverse.

''')
# mmr

df_mmr.merge(df_jobs, left_on='job_id', right_on='jobID')[:30]
# sim

df_rec_top100[df_rec_top100['user_id']=='1'].merge(df_jobs, left_on='job_id', right_on='jobID')[:30]

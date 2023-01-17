import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
market_train_df, news_train_df = env.get_training_data()
import missingno

def plot_miss(df):
    missingno.matrix(df)
    plt.show()
    
plot_miss(market_train_df)
def plot_distributions(df, labels, figsize, h, w):
    i=0
    fig = plt.figure(figsize=figsize)
    for l in labels:
        i+=1
        fig.add_subplot(h,w,i)
        plt.tight_layout()
        plt.title(f'{l}, mean = {df[l].mean():.2f}')
        lq = df[l].quantile(0.01)
        uq = df[l].quantile(0.99)
        feature = df[l].dropna()
        plt.hist(feature[(feature > lq)&(feature < uq)], density=True, bins=100, alpha=0.7)
    plt.show()
news_labels = ['wordCount', 'sentenceCount', 'companyCount', 'bodySize', 'urgency',
              'firstMentionSentence', 'relevance', 'sentimentClass',
              'sentimentNegative', 'sentimentNeutral', 'sentimentPositive',
              'sentimentWordCount', 'noveltyCount12H', 'noveltyCount24H',
              'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D', 'volumeCounts12H',
              'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D']

plot_distributions(news_train_df[news_labels], news_labels, figsize=(20,25), h=6, w=4)
market_labels = ['returnsClosePrevMktres10', 'returnsClosePrevMktres1']

plot_distributions(market_train_df[market_labels], market_labels, figsize=(15,5), h=1, w=3)
market_drop_features = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 
    'returnsOpenPrevRaw1','returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 
    'returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']
news_drop_features = ['urgency', 'takeSequence','provider', 'subjects', 
    'audiences','sentimentClass', 'headlineTag', 'sourceTimestamp', 'firstCreated', 
    'sourceId', 'headline', 'marketCommentary', 'assetCodes']
sem_labels = ['sentimentPositive', 'sentimentNegative', 'sentimentNeutral']
from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters = 9, init = 'k-means++')
semant_kmeans = kmeans.fit_predict(news_train_df[sem_labels].values)
semant_kmeans = np.reshape(semant_kmeans, (semant_kmeans.shape[0], 1))
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categories='auto')
kmean_dummies = onehotencoder.fit_transform(semant_kmeans).toarray()

def dummies_concat(df, dummies, pref):
    dummies = pd.DataFrame(dummies, index=df.index)#dummies[:, 1:]
    dummies.columns = [pref + str(x) for x in dummies.columns]
    return pd.concat([df, dummies], axis=1)
def corr_plot(df):
    corr = df.corr()

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()
num_labels = ['noveltyCount12H', 'noveltyCount24H','noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D',
             'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D',
             'wordCount', 'sentenceCount', 'companyCount', 'bodySize', 'sentimentWordCount', 'relevance',
             'firstMentionSentence']

corr_plot(news_train_df[num_labels])
def plot_relevance(s_relevance):
    plt.title(f'Relevance < 1, {sum(news_train_df["relevance"]<1)/news_train_df.shape[0]*100:.2f} % of data')
    plt.hist(s_relevance[s_relevance<1], bins=100)
    plt.show()
    plt.title(f'Relevance = 1, {sum(news_train_df["relevance"] == 1)/news_train_df.shape[0]*100:.2f} % of data')
    plt.hist(s_relevance[s_relevance<1], bins=100)
    plt.hist(s_relevance[s_relevance==1], bins=100)
    plt.show()
plot_relevance(news_train_df['relevance'])
def is_one(x):
    return 1 if x == 1 else 0

def is_positive(x):
    return 1 if x > 0 else 0

def is_abnormal(x, threshold):
    return 1 if np.abs(x) > threshold else 0
def max_pooling(df, features, index, agg_method):
    agg_methods = {k: agg_method for k in features}
    return df.groupby(index).agg(agg_methods).reset_index()
def state_reduce(market, news, kmean_dummies):
    # check positive return
    market['returnsClosePrevMktres1'] = market['returnsClosePrevMktres1'].apply(is_positive)
    market['returnsClosePrevMktres10'] = market['returnsClosePrevMktres10'].apply(is_positive)
    # check abnormal return
    std = market['returnsClosePrevMktres1'].std()
    market['abnormalClosePrevMktres1'] = market['returnsClosePrevMktres1'].apply(lambda x: is_abnormal(x, 2.5*std))
    # check relevance is equal to one
    news['relevance'] = news['relevance'].apply(is_one)
    # drop features, which unusefull for us
    market = market.drop(market_drop_features, axis=1)
    news = news.drop(news_drop_features + sem_labels + num_labels, axis=1)
    # add semant clusters
    news = dummies_concat(news, kmean_dummies, 'sem_cl_')
    # choose features that will be in state
    bin_features = [l for l in news.columns if l not in ['time', 'assetName']]
    # truncat date for merging data
    market['date'] = market['time'].dt.round('d') 
    news['date'] = news['time'].dt.round('d')
    # pooling news in one date
    news = max_pooling(news,  bin_features, ['date', 'assetName'], 'max')
    # merging data and fill NA
    market_train_df = pd.merge(market, news, how='left', on=['date', 'assetName']).fillna(2)
    # reduce state features to str state id
    state_features = bin_features + ['returnsClosePrevMktres1', 'returnsClosePrevMktres10', 'abnormalClosePrevMktres1']
    state = market_train_df[state_features[0]].astype(int).astype(str)
    for col in state_features[1:]:
        state += market_train_df[col].astype(int).astype(str)
    market_train_df['state'] = state
    
    return market_train_df.drop(state_features + ['date', 'assetName'], axis=1)
%%time

market_train_df = state_reduce(market_train_df, news_train_df, kmean_dummies)
market_train_df.head()
print('Count of unique states in train data', len(market_train_df['state'].unique()))
def fit_state_table(df):
    # positive return observations
    pos_reward = df[df['returnsOpenNextMktres10'] > 0]
    # positive return probability
    pos_reward_prob = pos_reward.shape[0] / df.shape[0]
    # probability of each state
    state_probs = train.groupby(['state']).size() / df.shape[0]
    # probability of each state with positive return
    pos_state_probs = pos_reward.groupby(['state']).size() / pos_reward.shape[0]
    # concat all of this to one table
    state_table = pd.concat([state_probs, pos_state_probs], axis = 1, 
                            keys=['state_prob', 'pos_state_probs'], sort=False).fillna(0)
    
    state_table.index.name = 'state'
    
    state_table['pos_return_prob'] = state_table['pos_state_probs'] * pos_reward_prob / state_table['state_prob']
    
    return state_table
train, test = np.split(market_train_df[['time', 'universe', 'state', 'returnsOpenNextMktres10']], 
                       [int(.8*len(market_train_df))])
print(f'State coverage: { test["state"].isin(train["state"]).sum() / len(test) }')
#test = test[test["state"].isin(train["state"])]
state_table = fit_state_table(train)
state_table.head()
def max_prob_prediction(df, st):
    df = pd.merge(df, st[['pos_return_prob']], how='left', on=['state']).fillna(0)
    return df['pos_return_prob'].apply(lambda p: p if p >= 0.5 else p-1).tolist()

def diff_prob_prediction(df, st):
    df = pd.merge(df, st[['pos_return_prob']], how='left', on=['state']).fillna(0)
    return df['pos_return_prob'].apply(lambda p: 2*p - 1).tolist()

def all_in_prob_prediction(df, st):
    df = pd.merge(df, st[['pos_return_prob']], how='left', on=['state']).fillna(0)
    return df['pos_return_prob'].apply(lambda p: 1 if p >= 0.5 else -1).tolist()
max_pred = max_prob_prediction(test, state_table)
diff_pred = diff_prob_prediction(test, state_table)
all_in_pred = all_in_prob_prediction(test, state_table)
max_rewards = test['returnsOpenNextMktres10'] * max_pred
diff_rewards = test['returnsOpenNextMktres10'] * diff_pred
all_in_rewards = test['returnsOpenNextMktres10'] * all_in_pred
def score(r):
    return np.mean(r)/np.std(r)
    
fig = plt.figure(figsize=(14,7))
fig.add_subplot(1,2,1)
plt.tight_layout()
plt.title(f'Rewards')
plt.hist(max_rewards.tolist(), bins=100, alpha=0.5,
         label=f'MAX m={np.mean(max_rewards):.5f}, std={np.std(max_rewards):.5f}')
plt.hist(diff_rewards.tolist(), bins=100, alpha=0.5,
         label=f'DIFF m={np.mean(diff_rewards):.5f}, std={np.std(diff_rewards):.5f}')
plt.hist(all_in_rewards.tolist(), bins=100, alpha=0.5,
         label=f'ALL_IN m={np.mean(all_in_rewards):.5f}, std={np.std(all_in_rewards):.5f}')
plt.legend()
fig.add_subplot(1,2,2)
plt.tight_layout()
plt.title(f'Total reward')
plt.plot(np.cumsum(max_rewards), 
         label=f'MAX sum={np.sum(max_rewards):.2f}, score={score(max_rewards):.4f}')
plt.plot(np.cumsum(diff_rewards), 
         label=f'DIFF sum={np.sum(diff_rewards):.2f}, score={score(diff_rewards):.4f}')
plt.plot(np.cumsum(all_in_rewards), 
         label=f'ALL_IN sum={np.sum(all_in_rewards):.2f}, score={score(all_in_rewards):.4f}')
plt.legend()
plt.show()
preddays = env.get_prediction_days()
for marketdf, newsdf, predtemplatedf in preddays:
    
    # cpredict semant clusters
    pred_kmeans = kmeans.predict(newsdf[sem_labels].values)
    pred_kmeans = np.reshape(pred_kmeans, (pred_kmeans.shape[0], 1))
    
    # encode semant clusters
    pred_dummies = onehotencoder.transform(pred_kmeans).toarray()
    
    # merge and pool data
    states = state_reduce(marketdf, newsdf, pred_dummies)
    
    # predict confidence
    preds = max_prob_prediction(states, state_table)
    
    #prediction
    predsdf = pd.DataFrame({'ast':states['assetCode'],'conf':preds})
    predtemplatedf.loc[predtemplatedf['assetCode'].isin(predsdf.ast), 'confidenceValue'] = predsdf['conf'].values
    
    env.predict(predtemplatedf)

env.write_submission_file()
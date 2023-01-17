# basic imports

import string

import re

import gc

from collections import Counter



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import colors

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import pickle



from nltk.tokenize import sent_tokenize, word_tokenize

from nltk import pos_tag

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm._tqdm_notebook import tqdm_notebook as tqdm; tqdm.pandas()



from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from sklearn.decomposition import TruncatedSVD



# dataframe options to display the whole comments

pd.set_option('display.max_colwidth', -1)



# extra config to have better visualization

sns.set(

    style='whitegrid',

    palette='coolwarm',

    rc={'grid.color' : '.96'}

)

plt.rcParams['font.size'] = 12

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['axes.labelweight'] = 'bold'

plt.rcParams['axes.titlesize'] = 20

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12

plt.rcParams['legend.fontsize'] = 14

plt.rcParams['figure.titlesize'] = 30

plt.rcParams["figure.titleweight"] = 'bold'
# pandas dataframe background gradient to consider all rows and columns.

# By default, background gradient styling in pandas only consider by column.

# function taken from: https://stackoverflow.com/questions/38931566/pandas-style-background-gradient-both-rows-and-columns

def background_gradient(s, m, M, cmap='PuBu', low=0, high=1):

    rng = M - m

    norm = colors.Normalize(m - (rng * low),

                            M + (rng * high))

    normed = norm(s.values)

    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]

    return ['background-color: %s' % color for color in c]
# data loading

# use only training set

train = pd.read_csv('../input/train.csv')

train.columns
toxic_subtypes = [

    'severe_toxicity',

    'obscene',

    'threat',

    'insult',

    'identity_attack',

    'sexual_explicit'

]



identity_attrs = [

    'asian', 'atheist', 'bisexual',

    'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',

    'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',

    'jewish', 'latino', 'male', 'muslim', 'other_disability',

    'other_gender', 'other_race_or_ethnicity', 'other_religion',

    'other_sexual_orientation', 'physical_disability',

    'psychiatric_or_mental_illness', 'transgender', 'white',

]



identity_attrs_group = {

    'gender': ['female', 'male', 'transgender', 'other_gender'],

    'race': ['asian', 'black', 'jewish', 'latino', 'white', 'other_race_or_ethnicity'],

    'religion': ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'other_religion'],

    'sexual_orientation': ['bisexual', 'heterosexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation'],

    'disability': ['intellectual_or_learning_disability', 'physical_disability', 'psychiatric_or_mental_illness', 'other_disability']

}
# create a 0 or 1 column and see the proportion

train['is_toxic'] = train['target'] >= 0.5

toxic_count = train['is_toxic'].value_counts()

toxic_prop = toxic_count / len(train['is_toxic'])

print("There are {:,} ({:.2f}%) toxic comments out of {:,} comments in the dataset".format(

    toxic_count[True],

    toxic_count[True] * 100 / len(train['is_toxic']),

    len(train['is_toxic'])

))
fig, ax = plt.subplots(figsize=(12, 7.5))

_ = sns.kdeplot(train['target'], shade=True, ax=ax)

_ = ax.set(xlabel='Annotator Toxicity Agreement', ylabel='Density')
train['log_toxicity_annotator_count'] = np.log10(train['toxicity_annotator_count'])

fig, ax = plt.subplots(figsize=(12, 7.5))

_ = train['log_toxicity_annotator_count'].hist(bins=15, density=True, ax=ax)

_ = ax.set(xlabel='log10(annotator count)', ylabel='Density')
train['many_annotators'] = train['log_toxicity_annotator_count'] >= 1.5



print("There are {:,} comments with number of annotators below 10^1.5 and {:,} above 10^1.5.".format(

    len(train[train['log_toxicity_annotator_count'] < 1.5]),

    len(train[train['log_toxicity_annotator_count'] >= 1.5])

))



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))



_ = sns.kdeplot(

    train[~train['many_annotators']]['target'], 

    shade=True,

    ax=ax1

)

_ = ax1.set_title('number of annotators < 10^1.5')

_ = ax1.set_ylabel('density')

_ = ax1.set_ylabel('#annotators')



_ = sns.kdeplot(

    train[train['many_annotators']]['target'], 

    shade=True,

    ax=ax2

)

_ = ax2.set_title('number of annotators >= 10^1.5')

_ = ax2.set_ylabel('density')

_ = ax2.set_ylabel('#annotators')

subtypes_count = (train[toxic_subtypes] > 0.5).sum(axis=0).sort_values(ascending=False)

subtypes_prop = np.round(subtypes_count * 100 / len(train), 2)

fig, ax = plt.subplots(figsize=(12, 7.5))

_ = sns.barplot(x=subtypes_count.index, y=subtypes_count.values, ax=ax)

_ = ax.set_title('Toxic Subtypes Distribution (with % of all comments)')

_ = ax.set_xlabel('Toxic Subtypes')

_ = ax.set_ylabel('#Comments')



for p, label in zip(ax.patches, subtypes_prop.values):

    ax.annotate("{:.2f}%".format(label), (p.get_x()+0.275, p.get_height()+500))
subtype_examples = []

for subtype_col in toxic_subtypes:

    comment = train[[subtype_col, 'toxicity_annotator_count', 'comment_text']].sort_values(

        by=[subtype_col, 'toxicity_annotator_count'], ascending=False).iloc[0]

    subtype_examples.append({

        'subtype' : subtype_col,

        'comment' : comment['comment_text'],

        'subtype_toxicity_level' : comment[subtype_col],

        'num_annotators' : comment['toxicity_annotator_count']

    })

subtype_examples_df = pd.DataFrame(subtype_examples).set_index('subtype')

subtype_examples_df
corr = train[toxic_subtypes + ['target']].corr()

corr.style.apply(background_gradient, m=corr.min().min(), M=corr.max().max()).set_precision(2)
identity_df = train[identity_attrs].dropna(axis=0, how='all')

print("There are {:,} ({:.2f}% of all training set) identity-labelled comments, out of which {:,} ({:.2f}%) are not identity offensive.".format(

    len(identity_df),

    len(identity_df) * 100 / len(train),

    len(identity_df[np.sum(identity_df, axis=1) == 0]),

    len(identity_df[np.sum(identity_df, axis=1) == 0]) * 100 / len(identity_df)

))
identity_count = (identity_df > 0.5).sum(axis=0).sort_values(ascending=False)

identity_prop = np.round(identity_count * 100 / len(identity_df), 2)

fig, ax = plt.subplots(figsize=(12, 7.5))

_ = sns.barplot(x=identity_count.values, y=identity_count.index, ax=ax)

_ = ax.set_title('Identity Distribution (with % of all identity-labeled comments)')

_ = ax.set_xlabel('Identity Labels')

_ = ax.set_ylabel('#Comments')



for p, label in zip(ax.patches, identity_prop.values):

    ax.annotate("{:.2f}%".format(label), (p.get_width(), p.get_y()+0.6))
identity_group_count = pd.Series(

    dict(

        (g, np.sum(identity_count[identity_count.index.isin(identity_attrs_group[g])])) 

        for g in identity_attrs_group)

).sort_values(ascending=False)

identity_group_prop = np.round(identity_group_count * 100 / len(identity_df), 2)

fig, ax = plt.subplots(figsize=(12, 7.5))

_ = sns.barplot(x=identity_group_count.index, y=identity_group_count.values, ax=ax)

_ = ax.set_title('Identity Distribution (with % of all comments w/ identity data)')

_ = ax.set_xlabel('Identity Types')

_ = ax.set_ylabel('#Comments')



for p, label in zip(ax.patches, identity_group_prop.values):

    ax.annotate("{:.2f}%".format(label), (p.get_x()+0.275, p.get_height()+500))
identity_group_df = pd.DataFrame()

for g in identity_attrs_group:

    identity_group_df[g] = np.max(identity_df[identity_attrs_group[g]], axis=1)

identity_group_w_target_df = identity_group_df.join(train['target'], how='left')

corr = identity_group_w_target_df.corr()

corr.style.apply(background_gradient, m=corr.min().min(), M=corr.max().max()).set_precision(2)
identity_group_w_subtypes_df = identity_group_df.join(train[toxic_subtypes], how='left')

corr = identity_group_w_subtypes_df.corr()

corr = corr.loc[toxic_subtypes, list(identity_attrs_group.keys())]

corr.style.apply(background_gradient, m=corr.min().min(), M=corr.max().max()).set_precision(2)
def plot_cdf(ax, df, col, xlabel):

    _ = ax.hist(df[(df['target'] >= 0.5) & (df[col] < df[col].quantile(.99))][col], 200, 

                               density=True, histtype='step', color='red',

                               cumulative=True, label='toxic')

    _ = ax.hist(df[(df['target'] < 0.5) & (df[col] < df[col].quantile(.99))][col], 200,

                               density=True, histtype='step', color='blue',

                               cumulative=True, label='non-toxic')

    _ = ax.legend(loc='upper left')

    _ = ax.set_xlabel(xlabel)

    _ = ax.set_ylabel('Proportion')

    

    return ax
train['char_length'] = train['comment_text'].progress_apply(lambda c: len(c))

train['tokenized_comment'] = train['comment_text'].progress_apply(

    lambda c: [t.lower() for t in re.split("[\s\-—]+", c.translate(str.maketrans('', '', string.punctuation))) if len(t) > 0]

)

train['num_tokens'] = train['tokenized_comment'].progress_apply(lambda c: len(c))

train['average_token_length'] = train['tokenized_comment'].progress_apply(lambda c: np.mean([len(t) for t in c]) if len(c) > 0 else 0)

train['comment_sentences'] = train['comment_text'].progress_apply(lambda c: sent_tokenize(c))

train['number_of_sentences'] = train['comment_sentences'].progress_apply(lambda s: len(s))

train['capital_letters_prop'] = train['comment_text'].progress_apply(lambda c: sum(1 for i in c if i.isupper()) / len(c))

train['non_alphanumeric_prop'] = train['comment_text'].progress_apply(lambda c: sum(1 for t in c if not t.isalnum()) / len(c))
fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True, figsize=(18, 18))

plt.suptitle('Lexical CDF')



_ = plot_cdf(axs[0,0], train, 'char_length', 'Number of Characters')

_ = plot_cdf(axs[0,1], train, 'num_tokens', 'Number of Tokens')

_ = plot_cdf(axs[1,0], train, 'average_token_length', 'Average Token Length')

_ = plot_cdf(axs[1,1], train, 'number_of_sentences', 'Number Of Sentences')

_ = plot_cdf(axs[2,0], train, 'capital_letters_prop', 'Capital Letters Proportion')

_ = plot_cdf(axs[2,1], train, 'non_alphanumeric_prop', 'Non-Alphanumeric Proportion')
n = len(train[train['target'] >= 0.5])

train_sample = pd.concat([train[train['target'] >= 0.5].sample(n=n, random_state=1336), train[train['target'] < 0.5].sample(n=2*n, random_state=1337)])

del identity_df

del train

_ = gc.collect()



n_train_sample = len(train_sample)

print("The number of samples: {:,}".format(n_train_sample))
train_sample_pos = train_sample.join(

    train_sample['tokenized_comment'].progress_apply(

        lambda c: pd.Series(Counter('POS_' + p[:2] + '_prop' for w,p in pos_tag(c))) / len(c)

    )

)
pos_columns = ['POS_JJ_prop', 'POS_NN_prop', 'POS_IN_prop',

       'POS_PR_prop', 'POS_VB_prop', 'POS_CC_prop', 'POS_MD_prop',

       'POS_RB_prop', 'POS_TO_prop', 'POS_DT_prop', 'POS_WD_prop',

       'POS_EX_prop', 'POS_WP_prop', 'POS_CD_prop', 'POS_WR_prop',

       'POS_PD_prop', 'POS_RP_prop', 'POS_UH_prop', 'POS_FW_prop',

       'POS_\'\'_prop', 'POS_PO_prop', 'POS_$_prop']

train_sample_pos[pos_columns + ['target']].corr()['target'].sort_values(ascending=False)[1:]
fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(18, 12))

plt.suptitle('POS CDF')



_ = plot_cdf(axs[0,0], train_sample_pos, 'POS_PR_prop', 'Pronoun Proportion')

_ = plot_cdf(axs[0,1], train_sample_pos, 'POS_PD_prop', 'Predeterminer Proportion')

_ = plot_cdf(axs[1,0], train_sample_pos, 'POS_JJ_prop', 'Adjectives Proportion')

_ = plot_cdf(axs[1,1], train_sample_pos, 'POS_RP_prop', 'Particles Proportion')
fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(18, 12))

plt.suptitle('POS CDF')



_ = plot_cdf(axs[0,0], train_sample_pos, 'POS_PO_prop', 'Possessive Ending Proportion')

_ = plot_cdf(axs[0,1], train_sample_pos, 'POS_CD_prop', 'Digits Proportion')

_ = plot_cdf(axs[1,0], train_sample_pos, 'POS_UH_prop', 'Interjection Proportion')

_ = plot_cdf(axs[1,1], train_sample_pos, 'POS_IN_prop', 'Preposition Proportion')
stemmer = SnowballStemmer("english")

train_sample['stemmed_comment'] = train_sample['tokenized_comment'].progress_map(lambda c: ' '.join([stemmer.stem(t) for t in c]))

comment_df = train_sample[['comment_text', 'stemmed_comment', 'toxicity_annotator_count', 'target']].sample(frac=1, random_state=1338)

del train_sample_pos

del train_sample

_ = gc.collect()
# constructing TF-IDF term-weighting vocabulary

# Only words that occur in at least 50 comments are included

vectorizer = TfidfVectorizer(min_df=50, max_df=.15, ngram_range=(1, 2))

train_n = int(0.1 * n_train_sample)

X_train = vectorizer.fit_transform(comment_df[:train_n]['stemmed_comment'])

X_test = vectorizer.transform(comment_df[train_n:]['stemmed_comment'])

y_train = comment_df[:train_n]['target'] >= 0.5

y_test = comment_df[train_n:]['target'] >= 0.5

print("Number of vocabulary: {:,}".format(len(vectorizer.get_feature_names())))
avg_tfidf = np.asarray(X_train.mean(axis=0)).ravel().tolist()

weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'avg_tfidf': avg_tfidf})

weights_df.sort_values(by='avg_tfidf', ascending=False).head(25)
def informative_features(vectorizer, clf, n=20):

    feature_names = vectorizer.get_feature_names()

    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))

    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])

    for (coef_1, fn_1), (coef_2, fn_2) in top:

        print("\t{:8.4f} * {:15}\t\t{:8.4f} * {:15}".format(coef_1, fn_1, coef_2, fn_2))
nb_model = MultinomialNB()

nb_model.fit(X_train, y_train, sample_weight=list(y_train * 0.5 + 0.5))

y_pred_nb = nb_model.predict(X_test)

y_prob_nb = nb_model.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(y_test, y_prob_nb)

fpr, tpr, threshold = roc_curve(y_test, y_prob_nb)

print(confusion_matrix(y_test, y_pred_nb))
fig, ax = plt.subplots(figsize=(12, 7.5))

_ = ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

_ = ax.legend(loc = 'lower right')

_ = ax.plot([0, 1], [0, 1],'r--')

_ = ax.set_title('Receiver Operating Characteristic')

_ = ax.set_ylabel('True Positive Rate')

_ = ax.set_xlabel('False Positive Rate')
informative_features(vectorizer, nb_model)
pred_df = pd.DataFrame()

pred_df['true'] = y_test.astype(int)

pred_df['prob'] = y_prob_nb

fp_indices = pred_df[(pred_df['true'] == 0) & (pred_df['prob'] >= 0.5)]['prob'].sort_values(ascending=False).index

tp_indices = pred_df[(pred_df['true'] == 0) & (pred_df['prob'] < 0.5)]['prob'].index

comment_df.loc[fp_indices[:25]][['comment_text', 'target', 'toxicity_annotator_count']].sort_values(['target'])
import os

import pandas_profiling as pp

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')



import datetime

import pandas_profiling as pp



from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from wordcloud import WordCloud

from collections import Counter

from nltk.corpus import stopwords

from nltk.util import ngrams



from sklearn.feature_extraction.text import TfidfVectorizer

stop = set(stopwords.words('russian'))



import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold



kf = KFold(n_splits=5)
df_train = pd.read_csv("../input/train.csv")

df_test  = pd.read_csv("../input/test.csv")

df_sub   = pd.read_csv("../input/sample_submission.csv")

df_periods_train = pd.read_csv("../input/periods_train.csv")
df_train.head(5)
df_train.info()
df_train.describe(include='all')
df_periods_train.head(5)
df_periods_train.info()
df_periods_train.describe(include='all')
pp.ProfileReport(df_train)
df_train["activation_date"] = pd.to_datetime(df_train["activation_date"])

df_train["date"]            = df_train["activation_date"].dt.date

df_train["weekday"]         = df_train["activation_date"].dt.weekday

df_train["day"]             = df_train["activation_date"].dt.day



count_by_date_train         = df_train.groupby("date")["deal_probability"].count()

mean_by_date_train          = df_train.groupby("date")["deal_probability"].mean()



df_test["activation_date"]  = pd.to_datetime(df_test["activation_date"])

df_test["date"]             = df_test["activation_date"].dt.date

df_test["weekday"]          = df_test["activation_date"].dt.weekday

df_test["day"]              = df_test["activation_date"].dt.day



count_by_date_test          = df_test.groupby('date')['item_id'].count()
fig, (ax1, ax3) = plt.subplots(figsize=(26, 8), 

                              ncols=2,

                              sharey=True)

count_by_date_train.plot(ax=ax1, 

                        legend=False,

                        label="Ads Count")



ax2 = ax1.twinx()



mean_by_date_train.plot(ax=ax2,

                       color="g",

                       legend=False,

                       label="Mean deal_probability")



ax2.set_ylabel("Mean deal_probabilit", color="g")



count_by_date_test.plot(ax=ax3,

                       color="r",

                       legend=False,

                       label="Ads count test")



plt.grid(False)



ax1.title.set_text("Trends of deal_probability and number of ads")

ax3.title.set_text("Trends of number of ads for test data")

ax1.legend(loc=(0.8, 0.35))

ax2.legend(loc=(0.8, 0.2))

ax3.legend(loc="upper right")
fig, ax1 = plt.subplots(figsize=(16, 8))



plt.title("Ads Count and Deal Probability by day of the week")



sns.countplot(x   = "weekday",

             data = df_train,

             ax   = ax1)



ax1.set_ylabel("Ads Count", color="b")



plt.legend(["Projects Count"])



ax2 = ax1.twinx()



sns.pointplot(x   = "weekday",

             y    = "deal_probability",

             data = df_train,

             ci   = 99,

             ax   = ax2,

             color = "black")



ax2.set_ylabel("deal_probability", color="g")

plt.legend(["deal_probability"], loc=(0.875, 0.9))

plt.grid(False)

a = df_train.groupby(["parent_category_name", "category_name"]).agg({"deal_probability": ["mean", "count"]}).reset_index().sort_values([("deal_probability", "mean")], ascending=False).reset_index(drop=True)



a
city_ads = df_train.groupby("city").agg({"deal_probability": ["mean", "count"]}).reset_index().sort_values([("deal_probability", "mean")], ascending=False).reset_index(drop=True)



print("There are {0} cities in total".format(len(df_train.city.unique())))



print("There are {1} cities with more than {0} ads".format(100, city_ads[city_ads["deal_probability"]["count"] > 100].shape[0]))



print('There are {1} cities with more that {0} ads.'.format(1000, city_ads[city_ads['deal_probability']['count'] > 1000].shape[0]))



print('There are {1} cities with more that {0} ads.'.format(10000, city_ads[city_ads['deal_probability']['count'] > 10000].shape[0]))
city_ads[city_ads["deal_probability"]["count"] > 1000].head()
city_ads[city_ads['deal_probability']['count'] > 1000].tail()
print("Лабинск")



df_train.loc[df_train.city == "Лабинск"].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head(5)
print('Миллерово')

df_train.loc[df_train.city == 'Миллерово'].groupby('category_name').agg({'deal_probability': ['mean', 'count']}).reset_index().sort_values([('deal_probability', 'count')], ascending=False).reset_index(drop=True).head()
plt.hist(df_train["deal_probability"]);

plt.title("deal_probability");
text = ' '.join(df_train["title"].values)

wordCloud = WordCloud(max_font_size = None,

                      stopwords = stop,

                      background_color = "white",

                      width = 1200,

                      height = 1000).generate(text)



plt.figure(figsize=(12, 8))

plt.imshow(wordCloud)

plt.title('Top words for title')

plt.axis("off")

plt.show()
df_train["description"] = df_train["description"].apply(

    lambda x: str(x).replace('/\n', ' ').replace('\xa0', ' ')

)
text = ' '.join(df_train['description'].values)

text = [i for i in ngrams(text.lower().split(), 3)]

print('Common trigrams.')

Counter(text).most_common(40)
df_train[df_train.description.str.contains('↓')]['description'].head(10).values
df_train['has_image'] = 1

df_train.loc[df_train['image'].isnull(),'has_image'] = 0



print('There are {} ads with images. Mean deal_probability is {:.3}.'.format(len(df_train.loc[df_train['has_image'] == 1]), df_train.loc[df_train['has_image'] == 1, 'deal_probability'].mean()))
print('There are {} ads without images. Mean deal_probability is {:.3}.'.format(len(df_train.loc[df_train['has_image'] == 0]), df_train.loc[df_train['has_image'] == 0, 'deal_probability'].mean()))
plt.scatter(df_train.item_seq_number, df_train.deal_probability, label="item_seq_number vs deal_probability");



plt.xlabel("item_seq_number");

plt.ylabel("deal_probability");
df_train["params"] = df_train["param_1"].fillna('') + ' ' + df_train["param_2"].fillna('') + ' ' + df_train["param_3"].fillna('')



df_train["params"] = df_train["params"].str.strip()



text = ' '.join(df_train["params"].values)

text = [i for i in ngrams(text.lower().split(), 3)]



print("common trigrams")



Counter(text).most_common(40)
sns.set(rc = {'figure.figsize': (15, 8)})



df_train_ = df_train[df_train.price.isnull() == False]

df_train_ = df_train.loc[df_train.price < 100000.0]



sns.boxplot(x = "parent_category_name",

           y = "price",

           hue = "user_type",

           data = df_train_)



plt.title("Price by parent gategory and user type")

plt.xticks(rotation = "vertical")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
df_train["price"] = df_train.groupby(["city", "category_name"])["price"].apply(

    lambda x: x.fillna(x.median())

)



df_train["price"] = df_train.groupby(["region", "category_name"])["price"].apply(

    lambda x: x.fillna(x.median())

)



df_train["price"] = df_train.groupby(["category_name"])["price"].apply(

    lambda x: x.fillna(x.median())

)



plt.hist(df_train["price"]);
plt.hist(stats.boxcox(df_train["price"] + 1)[0]);
# lets transform the test in the same way as train



df_test["params"] = df_test["param_1"].fillna('') + ' ' + df_test["param_2"].fillna('') + ' ' + df_test["param_3"].fillna('')

df_test["params"] = df_test["params"].str.strip()



df_test["description"] = df_test["description"].apply( lambda x: str(x).replace("/\n", ' ').replace("\xa0", " "))



df_test["has_image"] = 1

df_test.loc[df_test["image"].isnull(), "has_image"] = 0



df_test["price"] = df_test.groupby(["city", "category_name"])["price"].apply(lambda x: x.fillna(x.median()))

df_test["price"] = df_test.groupby(["region", "category_name"])["price"].apply(lambda x: x.fillna(x.median()))

df_test["price"] = df_test.groupby(["category_name"])["price"].apply(lambda x: x.fillna(x.median()))



df_train["price"] = stats.boxcox(df_train.price + 1)[0]

df_test["price"]  = stats.boxcox(df_test.price + 1)[0]
df_train["user_price_mean"] = df_train.groupby("user_id")["price"].transform("mean")

df_train["user_ad_count"]   = df_train.groupby("user_id")["price"].transform("sum")



df_train["region_price_mean"]   = df_train.groupby("region")["price"].transform("mean")

df_train["region_price_median"] = df_train.groupby("region")["price"].transform("median")

df_train["region_price_max"]    = df_train.groupby("region")["price"].transform("max")



df_train["city_price_mean"]   = df_train.groupby("region")["price"].transform("mean")

df_train["city_price_median"] = df_train.groupby("region")["price"].transform("median")

df_train["city_price_max"]    = df_train.groupby("region")["price"].transform("max")



df_train["parent_category_name_price_mean"]   = df_train.groupby("parent_category_name")["price"].transform("mean")

df_train["parent_category_name_price_median"] = df_train.groupby("parent_category_name")["price"].transform("median")

df_train["parent_category_name_price_max"]    = df_train.groupby("parent_category_name")["price"].transform("max")

df_train["category_name_price_mean"]   = df_train.groupby("category_name")["price"].transform("mean")

df_train["category_name_price_median"] = df_train.groupby("category_name")["price"].transform("median")

df_train["category_name_price_max"]    = df_train.groupby("category_name")["price"].transform("max")



df_train["user_type_category_price_mean"]   = df_train.groupby(["user_type", "parent_category_name"])["price"].transform("mean")

df_train["user_type_category_price_median"] = df_train.groupby(["user_type", "parent_category_name"])["price"].transform("mean")

df_train["user_type_category_price_mean"]   = df_train.groupby(["user_type", "parent_category_name"])["price"].transform("mean")
df_test["user_price_mean"] = df_test.groupby("user_id")["price"].transform("mean")

df_test["user_ad_count"]   = df_test.groupby("user_id")["price"].transform("sum")



df_test["region_price_mean"]   = df_test.groupby("region")["price"].transform("mean")

df_test["region_price_median"] = df_test.groupby("region")["price"].transform("median")

df_test["region_price_max"]    = df_test.groupby("region")["price"].transform("max")



df_test["city_price_mean"]   = df_test.groupby("region")["price"].transform("mean")

df_test["city_price_median"] = df_test.groupby("region")["price"].transform("median")

df_test["city_price_max"]    = df_test.groupby("region")["price"].transform("max")



df_test["parent_category_name_price_mean"]   = df_test.groupby("parent_category_name")["price"].transform("mean")

df_test["parent_category_name_price_median"] = df_test.groupby("parent_category_name")["price"].transform("median")

df_test["parent_category_name_price_max"]    = df_test.groupby("parent_category_name")["price"].transform("max")

df_test["category_name_price_mean"]   = df_test.groupby("category_name")["price"].transform("mean")

df_test["category_name_price_median"] = df_test.groupby("category_name")["price"].transform("median")

df_test["category_name_price_max"]    = df_test.groupby("category_name")["price"].transform("max")



df_test["user_type_category_price_mean"]   = df_test.groupby(["user_type", "parent_category_name"])["price"].transform("mean")

df_test["user_type_category_price_median"] = df_test.groupby(["user_type", "parent_category_name"])["price"].transform("mean")

df_test["user_type_category_price_mean"]   = df_test.groupby(["user_type", "parent_category_name"])["price"].transform("mean")
def target_encode(trn_series = None,

                 tst_series  = None, 

                 target      = None,

                 min_samples_leaf = 1,

                 smoothing   = 1,

                 noise_level = 0):

    """

    

    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    

    # Apply average function to all target data

    prior = target.mean()

    

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return ft_trn_series, ft_tst_series

    
df_train['parent_category_name'], df_test['parent_category_name'] = target_encode(df_train['parent_category_name'], df_test['parent_category_name'], df_train['deal_probability'])



df_train['category_name'], df_test['category_name'] = target_encode(df_train['category_name'], df_test['category_name'], df_train['deal_probability'])



df_train['region'], df_test['region'] = target_encode(df_train['region'], df_test['region'], df_train['deal_probability'])

df_train['image_top_1'], df_test['image_top_1'] = target_encode(df_train['image_top_1'], df_test['image_top_1'], df_train['deal_probability'])



df_train['city'], df_test['city'] = target_encode(df_train['city'], df_test['city'], df_train['deal_probability'])



df_train['param_1'], df_test['param_1'] = target_encode(df_train['param_1'], df_test['param_1'], df_train['deal_probability'])

df_train['param_2'], df_test['param_2'] = target_encode(df_train['param_2'], df_test['param_2'], df_train['deal_probability'])

df_train['param_3'], df_test['param_3'] = target_encode(df_train['param_3'], df_test['param_3'], df_train['deal_probability'])
df_train.drop(['date', 'day', 'user_id'], axis=1, inplace=True)

df_test.drop(['date', 'day', 'user_id'], axis=1, inplace=True)
df_train["len_title"] = df_train["title"].apply(lambda x: len(x))

df_train["words_title"] = df_train["title"].apply(lambda x: len(x.split()))



df_train["len_description"] = df_train["description"].apply(lambda x: len(x))

df_train["words_description"] = df_train["description"].apply(lambda x: len(x.split()))



df_train["len_params"] = df_train["params"].apply(lambda x: len(x))

df_train["words_params"] = df_train["params"].apply(lambda x: len(x.split()))



df_train['symbol1_count'] = df_train['description'].str.count('↓')

df_train['symbol2_count'] = df_train['description'].str.count('\*')

df_train['symbol3_count'] = df_train['description'].str.count('✔')

df_train['symbol4_count'] = df_train['description'].str.count('❀')

df_train['symbol5_count'] = df_train['description'].str.count('➚')

df_train['symbol6_count'] = df_train['description'].str.count('ஜ')

df_train['symbol7_count'] = df_train['description'].str.count('.')

df_train['symbol8_count'] = df_train['description'].str.count('!')

df_train['symbol9_count'] = df_train['description'].str.count('\?')

df_train['symbol10_count'] = df_train['description'].str.count('  ')

df_train['symbol11_count'] = df_train['description'].str.count('-')

df_train['symbol12_count'] = df_train['description'].str.count(',')



df_test['len_title']         = df_test['title'].apply(lambda x: len(x))

df_test['words_title']       = df_test['title'].apply(lambda x: len(x.split()))

df_test['len_description']   = df_test['description'].apply(lambda x: len(x))

df_test['words_description'] = df_test['description'].apply(lambda x: len(x.split()))

df_test['len_params']        = df_test['params'].apply(lambda x: len(x))

df_test['words_params']      = df_test['params'].apply(lambda x: len(x.split()))



df_test['symbol1_count'] = df_test['description'].str.count('↓')

df_test['symbol2_count'] = df_test['description'].str.count('\*')

df_test['symbol3_count'] = df_test['description'].str.count('✔')

df_test['symbol4_count'] = df_test['description'].str.count('❀')

df_test['symbol5_count'] = df_test['description'].str.count('➚')

df_test['symbol6_count'] = df_test['description'].str.count('ஜ')

df_test['symbol7_count'] = df_test['description'].str.count('.')

df_test['symbol8_count'] = df_test['description'].str.count('!')

df_test['symbol9_count'] = df_test['description'].str.count('\?')

df_test['symbol10_count'] = df_test['description'].str.count('  ')

df_test['symbol11_count'] = df_test['description'].str.count('-')

df_test['symbol12_count'] = df_test['description'].str.count(',')
vectorizer = TfidfVectorizer(stop_words = stop, max_features = 6000)

vectorizer.fit(df_train["title"])



df_train_title = vectorizer.transform(df_train["title"])

df_test_title  = vectorizer.transform(df_test["title"])

df_train.drop(["title", "params", "description", "user_type", "activation_date"], axis=1, inplace=True)

df_test.drop(["title", "params", "description", "user_type", "activation_date"], axis=1, inplace=True)
pd.set_option('max_columns', 60)

df_train.head()
%%time



X_meta = np.zeros((df_train_title.shape[0], 1))

X_test_meta = []



for fold_i, (train_i, test_i) in enumerate(kf.split(df_train_title)):

    print(fold_i)

    model = Ridge()

    model.fit(df_train_title.tocsr()[train_i], df_train["deal_probability"][train_i])

    X_meta[test_i, :] = model.predict(df_train_title.tocsr()[test_i]).reshape(-1, 1)

    X_test_meta.append(model.predict(df_test_title))

    

X_test_meta = np.stack(X_test_meta)

X_test_meta_mean = np.mean(X_test_meta, axis=0)
X_full = csr_matrix(hstack([df_train.drop(['item_id', 'deal_probability', 'image'], axis=1), X_meta]))

X_test_full = csr_matrix(hstack([df_test.drop(['item_id', 'image'], axis=1), X_test_meta_mean.reshape(-1, 1)]))



X_train, X_valid, y_train, y_valid = train_test_split(X_full, df_train["deal_probability"], test_size=0.20, random_state=42)
def rmse(predictions, targets):

    return np.sqrt( ( (predictions - targets) ** 2).mean() )
# took parameters from this kernel:  https://www.kaggle.com/the1owl/beep-beep



params = {"learning_rate": 0.08,

          "max_depth": 8,

          "boosting": "gbdt",

          "objective": "regression",

          "metric": ["auc", "rmse"],

          "is_training_metric": True,

          "seed": 19,

          "num_leaves": 63,

          "feature_fraction": 0.9,

          "bagging_fraction": 0.8,

          "bagging_freq": 5

         }



model = lgb.train(params,

                 lgb.Dataset(X_train, label=y_train),

                 2000,

                 lgb.Dataset(X_valid, label=y_valid),

                 verbose_eval=50,

                 early_stopping_rounds=20)
pred = model.predict(X_test_full)



#clipping is necessary.

df_sub['deal_probability'] = np.clip(pred, 0, 1)

df_sub.to_csv('sub.csv', index=False)
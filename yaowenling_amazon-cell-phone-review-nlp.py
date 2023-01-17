# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Step 0: Load datasets

import pandas as pd

items = pd.read_csv("/kaggle/input/amazon-cell-phones-reviews/20190928-items.csv")

reviews = pd.read_csv("/kaggle/input/amazon-cell-phones-reviews/20190928-reviews.csv")
# 0.1 Items overview

print("The dataset contains {0[0]: .0f} rows and {0[1]: .0f} variables.".format(items.shape))

items.head()
items.describe(include="all")
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("ggplot")

# Brand distribution

ax = items.groupby("brand").count()["asin"].plot(kind="pie", 

                                                 figsize=(8, 5),

                                                 title="Number of Offerings grouped by Brand")

plt.show()
# Average rating per brand

ax = items.groupby("brand").mean()["rating"].sort_values().plot(kind="barh",

                                                                figsize=(8,5), 

                                                                title="Average rating per Brand")

plt.show()
# 0.2 Reviews overview

print("The dataset contains {0[0]: ,.0f} rows and {0[1]: .0f} variables.".format(reviews.shape))

reviews.head()
# 0.4 Link review data to item data bsed on common column "asin"

reviews = pd.merge(reviews, items, how="left", left_on="asin", right_on="asin")
# 0.5 Rename columns

reviews.rename(columns={"rating_x": "rating", "title_x": "title", "title_y": "item_title", "rating_y": "overall_rating"}, inplace=True)

reviews.head()
# 0.6 Convert string into datetime

from datetime import datetime

reviews["date"] = reviews["date"].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))

reviews["date"].head()
# 0.7 Truncate date column to month

reviews["month"] = reviews["date"].apply(lambda x: x.replace(day=1))

reviews["month"].head()
# 0.8 Plot reviews over time

ax = pd.pivot_table(reviews, 

                    index="month", 

                    columns="brand", 

                    values="asin", 

                    aggfunc="count", 

                    fill_value=0).plot.area(title="Monthly Number of Reviews per Brand", figsize=(10, 6))
# 0.9 Add posivity label

reviews["positivity"] = reviews["rating"].apply(lambda x: 1 if x>3 else(0 if x==3 else -1))
# Step 1: Preprocess review text

# 1.1 Define preprocess function

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

import string

stop = set(stopwords.words('english'))

punc = set(string.punctuation)

keywords = reviews["brand"].apply(lambda x: x.lower()).unique().tolist()

keywords.append("phone")

lemma = WordNetLemmatizer()

def clean_text(text):

    # Convert the text into lowercase

    text = text.lower()

    # Split into list

    wordList = text.split()

    # Remove punctuation

    wordList = ["".join(x for x in word if (x=="'")|(x not in punc)) for word in wordList]

    # Remove stopwords

    wordList = [word for word in wordList if word not in stop]

    # Remove other keywords

    wordList = [word for word in wordList if word not in keywords]

    # Lemmatisation

    wordList = [lemma.lemmatize(word) for word in wordList]

    return " ".join(wordList)

clean_text("I love reading books.")
# 1.2 Apply preprocess function to the dataframe

reviews["body"] = reviews["body"].astype("str")

reviews["clean_text"] = reviews["body"].apply(clean_text)
reviews["clean_text"].head().values
# Step 2: Create a wordcloud

# 2.1 Define word frequency function

def word_freq_dict(text):

    # Convert text into word list

    wordList = text.split()

    # Generate word freq dictionary

    wordFreqDict = {word: wordList.count(word) for word in wordList}

    return wordFreqDict

word_freq_dict("I love reading books. I love music.")
# 2.2 Create brand subsets

apple = reviews[reviews["brand"]=="Apple"].sort_values(by=["date"], ascending=False)

samsung = reviews[reviews["brand"]=="Samsung"].sort_values(by=["date"], ascending=False)

xiaomi = reviews[reviews["brand"]=="Xiaomi"].sort_values(by=["date"], ascending=False)
# 2.3 Initializer

from wordcloud import WordCloud, ImageColorGenerator



# Define a function to create a wordcloud from dictionary of word frequency

def wordcloud_from_frequency(word_freq_dict, title, figure_size=(10, 6)):

    wordcloud.generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=figure_size)

    plt.imshow(wordcloud)

    plt.axis("off")

    plt.title(title)

    plt.show()

    

# Define a function to plot top10 positive words and top10 negative words in a grouped bar plot (from dictionaries)

def topn_wordfreq_bar_both(pos_word_freq_dict, neg_word_freq_dict, pos_num_doc, neg_num_doc, topn, title, palette, height=6, aspect=2):

    # Transform positive word frequency into DF

    df_pos = pd.DataFrame.from_dict(pos_word_freq_dict, orient="index").sort_values(by=0, ascending=False).head(topn)

    df_pos.columns = ["frequency"]

    df_pos["frequency"] = df_pos["frequency"] / pos_num_doc

    df_pos["label"] = "Positive"

    # Transform negative word frequency into DF

    df_neg = pd.DataFrame.from_dict(neg_word_freq_dict, orient="index").sort_values(by=0, ascending=False).head(topn)

    df_neg.columns = ["frequency"]

    df_neg["frequency"] = df_neg["frequency"] / neg_num_doc

    df_neg["label"] = "Negative"

    # Append two dataframes

    df_append = df_pos.append(df_neg)

    df_append.reset_index(inplace=True)

    # Plot

    sns.catplot(x="index", y="frequency", hue="label", data=df_append, 

                kind="bar",

                palette=palette,

                height=height, aspect=aspect, 

                legend_out=False)

    plt.title(title)

    plt.show()
# 2.4 Plot wordclouds for latest 1000 reviews for Apple

apple_pos = " ".join(apple[apple["positivity"]==1]["clean_text"][0:1000])

apple_pos_word_freq = word_freq_dict(apple_pos)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Blues",

                      background_color="white")

wordcloud_from_frequency(apple_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Apple")
apple[apple["clean_text"].apply(lambda x: "new" in x)]["item_title"].value_counts().sort_values(ascending=True).tail(10).plot(kind="barh")

plt.title("Most reviews that mention 'new' are from renewed iPhone buyers")

plt.show()
apple["renewed"] = apple["item_title"].apply(lambda x: ("Renewed" in x) | ("Reburshied" in x))

print("{0: 0.1%} iPhones that were sold on Amazon are renewed/reburshied.".format(apple["renewed"].sum() / len(apple["renewed"])))
apple_neg = " ".join(apple[apple["positivity"]==-1]["clean_text"][0:1000])

apple_neg_word_freq = word_freq_dict(apple_neg)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Blues",

                      background_color="black")

wordcloud_from_frequency(apple_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Apple")
topn_wordfreq_bar_both(apple_pos_word_freq, apple_neg_word_freq, 

                       min(sum(apple["positivity"]==1), 1000), 

                       min(sum(apple["positivity"]==-1), 1000), 

                       10, 

                       "Top10 Frequent Words in Latest Positive and Negative Reviews for Apple", 

                       ["lightblue", "lightcoral"], 

                       height=6, aspect=2)
# 2.5 Plot wordclouds for latest 1000 reviews for Samsung

samsung_pos = " ".join(samsung[samsung["positivity"]==1]["clean_text"][0:1000])

samsung_pos_word_freq = word_freq_dict(samsung_pos)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Greens",

                      background_color="white")

wordcloud_from_frequency(samsung_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Samsung")
samsung_neg = " ".join(samsung[samsung["positivity"]==-1]["clean_text"][0:1000])

samsung_neg_word_freq = word_freq_dict(samsung_neg)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Greens",

                      background_color="black")

wordcloud_from_frequency(samsung_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Samsung")
topn_wordfreq_bar_both(samsung_pos_word_freq, samsung_neg_word_freq, 

                       min(sum(samsung["positivity"]==1), 1000), 

                       min(sum(samsung["positivity"]==-1), 1000), 

                       10, 

                       "Top10 Frequent Words in Latest Positive and Negative Reviews for Samsung", 

                       ["steelblue", "orange"], 

                       height=6, aspect=2)
# 2.6 Plot wordclouds for latest 1000 reviews for Xiaomi

xiaomi_pos = " ".join(xiaomi[xiaomi["positivity"]==1]["clean_text"][0:1000])

xiaomi_pos_word_freq = word_freq_dict(xiaomi_pos)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Oranges",

                      background_color="white")

wordcloud_from_frequency(xiaomi_pos_word_freq, "Most Frequent Words in the Latest 1000 Positive Reviews for Xiaomi")
xiaomi_neg = " ".join(xiaomi[xiaomi["positivity"]==-1]["clean_text"][0:1000])

xiaomi_neg_word_freq = word_freq_dict(xiaomi_neg)

wordcloud = WordCloud(width=5000, 

                      height=3000, 

                      max_words=200, 

                      colormap="Oranges",

                      background_color="black")

wordcloud_from_frequency(xiaomi_neg_word_freq, "Most Frequent Words in the Latest 1000 Negative Reviews for Xiaomi")
topn_wordfreq_bar_both(xiaomi_pos_word_freq, xiaomi_neg_word_freq, 

                       min(sum(xiaomi["positivity"]==1), 1000), 

                       min(sum(xiaomi["positivity"]==-1), 1000), 

                       10, 

                       "Top10 Frequent Words in Latest Positive and Negative Reviews for Xiaomi", 

                       ["darkgreen", "pink"], 

                       height=6, aspect=2)
# Step 3: Vectorization and Topic Modelling

# 3.1 Initialize TF-IDF vectorizer

import time

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.05, stop_words="english")
# 3.2 Initalize LDA model

from sklearn.decomposition import LatentDirichletAllocation

n_topics=10

lda = LatentDirichletAllocation(n_components=n_topics, 

                                max_iter=50, 

                                learning_method='online',

                                learning_offset=50.,

                                random_state=0)
# 3.3 Define a function to print LDA topics

def print_topn_words(model, feature_names, topn):

    for topic_idx, topic in enumerate(model.components_):

        message = "Topic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-topn - 1:-1]])

        print(message)

    print()
# 3.4 Run LDA model for Apple

t0 = time.time()

apple_tfidf = tfidf_vectorizer.fit_transform(apple["clean_text"])

apple_tfidf_feature_names = tfidf_vectorizer.get_feature_names()

lda.fit(apple_tfidf)

print("Below is the output from LDA model with {} topics (each includes Top10 words) for Apple.".format(n_topics))

print_topn_words(lda, apple_tfidf_feature_names, 10)

print("Done in %0.3fs." % (time.time() - t0))
# Run the model for Samsung

t0 = time.time()

samsung_tfidf = tfidf_vectorizer.fit_transform(samsung["clean_text"])

samsung_tfidf_feature_names = tfidf_vectorizer.get_feature_names()

lda.fit(samsung_tfidf)

print("Below is the output from LDA model with {} topics (each includes Top10 words) for Samsung.".format(n_topics))

print_topn_words(lda, samsung_tfidf_feature_names, 10)

print("Done in %0.3fs." % (time.time() - t0))
# Run the model for Xiaomi

t0 = time.time()

xiaomi_tfidf = tfidf_vectorizer.fit_transform(xiaomi["clean_text"])

xiaomi_tfidf_feature_names = tfidf_vectorizer.get_feature_names()

lda.fit(xiaomi_tfidf)

print("Below is the output from LDA model with {} topics (each includes Top10 words) for Xiaomi.".format(n_topics))

print_topn_words(lda, xiaomi_tfidf_feature_names, 10)

print("Done in %0.3fs." % (time.time() - t0))
# Step 4: Plot feature importance using XGBoost

# 4.1 for Apple

import xgboost as xgb

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(apple_tfidf, apple["positivity"])

featureImport = pd.DataFrame(xgb_clf.feature_importances_, index=apple_tfidf_feature_names)

featureImport.columns = ["Importance"]

featureImport.sort_values(["Importance"], ascending=True).tail(20).plot(kind="barh", figsize=(10, 6))

plt.title("XGBoost Relative Feature Importance (from all reviews for Apple)")

plt.show()
# Step 1: Filter English reviews

# 1.1 Add language labels (This part can take ~15 minutes)

# from langdetect import detect

# def lang_detect(text):

#     try:

#         return detect(text)

#     except:

#         return None

# import time

# start_time = time.time()

# reviews["lang"] = reviews["body"].apply(lang_detect)

# print("It takes %s seconds for the code to finish." % (time.time() - start_time))
# 1.2 Plot distribution of reviews into languages

# reviews["lang"].value_counts()[:10].plot(kind="barh", title="Number of Reviews grouped by Top10 Language")

# plt.show()
# 1.3 Only take English reviews

# reviews = reviews[reviews["lang"]=="en"]
# Step 2: Sentiment analysis using Vader

# 2.1 Load packages

# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# analyzer = SentimentIntensityAnalyzer()

# analyzer.polarity_scores("The weather is nice today.")
# 2.2 Create sentiment score columns (It takes roughly 5 minutes)

# start_time = time.time()

# reviews["body"] = reviews["body"].astype("str")

# reviews["sent_neg"] = reviews["body"].apply(lambda x: analyzer.polarity_scores(x)["neg"])

# reviews["sent_neu"] = reviews["body"].apply(lambda x: analyzer.polarity_scores(x)["neu"])

# reviews["sent_pos"] = reviews["body"].apply(lambda x: analyzer.polarity_scores(x)["pos"])

# reviews["sent_comp"] = reviews["body"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

# print("It takes %s seconds for the code to finish." % (time.time() - start_time))
# 2.3 Save the datasets into csv

# reviews.to_csv("reviews_with_sentiment_scores.csv")
# After the steps above we will get a pre-proceessed dataset

reviews_en = pd.read_csv("/kaggle/input/amazon-cell-phones-reviews-with-sentiment-scores/reviews_with_sentiment_scores.csv")
# 2.4 Plot the distribution of sentiment scores

plt.figure()



plt.subplot(2, 2, 1)

reviews_en["sent_neg"].hist(figsize=(10, 8), color="lightblue")

plt.title("Negative Sentiment Score")

plt.subplot(2, 2, 2)

reviews_en["sent_neu"].hist(figsize=(10, 8), color="grey")

plt.title("Neutral Sentiment Score")

plt.subplot(2, 2, 3)

reviews_en["sent_pos"].hist(figsize=(10, 8), color="lightgreen")

plt.title("Positive Sentiment Score")

plt.subplot(2, 2, 4)

reviews_en["sent_comp"].hist(figsize=(10, 8), color="lightcoral")

plt.title("Compound Sentiment Score")



plt.suptitle('Sentiment Analysis of Amazom Cell Phone Reviews', fontsize=12, fontweight='bold');



plt.show()
# 2.5 Check the correlation between sentiment score (compound) and rating

import numpy as np

import scipy.stats as stats

print("The correlation coefficient between sentiment score (compound) and rating is {0[0]: .4f} with a p-value of {0[1]: .4f}.".format(stats.pearsonr(reviews_en["rating"], reviews_en["sent_comp"])))

reviews_en.groupby("rating").mean()["sent_comp"].plot(kind="bar", figsize=(10, 6))

plt.title("Avg. Sentiment Score (Compound) per Rating")

plt.show()
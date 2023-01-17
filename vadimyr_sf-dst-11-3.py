import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
from tqdm.notebook import tqdm
from pandas_profiling import ProfileReport
from itertools import islice
from datetime import datetime
import pprint
import ast
from catboost.text_processing import Tokenizer
import pprint
import ast
from catboost.text_processing import Tokenizer
from collections import Counter
import cufflinks


cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

pp = pprint.PrettyPrinter(indent=4)


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


food_data_train = pd.read_csv("/kaggle/input/sf-dst-restaurant-rating/main_task.csv")
food_data_test = pd.read_csv("/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv")

world_cityes = pd.read_csv("/kaggle/input/world-cities/worldcities.csv")
country_population = pd.read_csv("/kaggle/input/pop-by-country/pop_by_country.csv")


food_data_train["sample"] = 1  # it's traning dataset
food_data_test["sample"] = 0  # it's test dataset
food_data_test["Rating"] = 0  # equal 0 for prediction

food_data = food_data_test.append(food_data_train, sort=False).reset_index(
    drop=True
)


pd.set_option("display.max_rows", 50)
pd.set_option("display.max_columns", 50)
display(food_data.head(5))
display(country_population.head(5))
# Cardinality характеризует уникальность данных.
# Высокая кардинальность - уникальные данные,
# низкая кардинальность - повторяющиеся данные.
# profile = ProfileReport(food_data, config_file="./input/config_dark.yaml")
# profile.to_file("project3.html")
index = food_data.index

number_of_rows = len(index)
uniq_vs_nan = pd.DataFrame(index=food_data.columns, columns=["uniq", "nulled"])

for col in tqdm(food_data.columns):
    # this is more easy way to countiing nulled values
    uniq_vs_nan.loc[col, "nulled"] = food_data[col].isnull().sum(axis=0)
    # try:
    #    nan_count = pd.notnull(food_data[col]).value_counts()[False]
    # except KeyError:  # нет значений nan
    #    nan_count = 0
    uniq_vs_nan.loc[col, "uniq"] = food_data[col].nunique()


fig = go.Figure(
    data=[
        go.Bar(
            name="Uniq values (abs)",
            x=uniq_vs_nan.index,
            y=uniq_vs_nan["uniq"],
        ),
        go.Bar(
            name="NaN values (abs)",
            x=uniq_vs_nan.index,
            y=uniq_vs_nan["nulled"],
        ),
    ]
)
# Change the bar mode
fig.update_layout(barmode="group")
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "uniq to NaN",
        "y": 0.85,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Values count",
    yaxis_title="Column name",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
# if the zero values are rather low, they may not be visible in the chart
# check all columns that have zero values.
display(uniq_vs_nan[uniq_vs_nan.nulled > 0])
# just other zero values checking method 
# duble check matter :)))
null_value_stats = food_data.isnull().sum(axis=0)
null_value_stats[null_value_stats != 0]
display(null_value_stats)
fig = go.Figure(
    data=[
        go.Bar(
            name="Rating",
            x=food_data["Rating"].value_counts().index,
            y=food_data["Rating"].value_counts(),
        )
    ]
)
# Change the bar mode
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Target var distribution",
        "y": 0.85,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Rating value",
    yaxis_title="Rating value count",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
# we have nan in 3 columns: "Cuisine Style", "Price Range", "Number of Reviews"
# don't sav info about 'Reviews' because it's only 2 nan values
# save information about nan's in dataset
def save_nan(param):
    return food_data[param].apply(lambda x: 1 if pd.isna(x) else 0)


food_data["NO Cuisine Style"] = save_nan("Cuisine Style")
food_data["NO Price Range"] = save_nan("Price Range")
food_data["NO Number of Reviews"] = save_nan("Number of Reviews")
food_data['Number of Reviews'].fillna(0, inplace=True)

# check
display(food_data[food_data["NO Cuisine Style"] == 1].head(5))
food_data.info()
food_data["Price Range"].value_counts()
# We have 17361 NaN's in "Price Range"
# Most popular value  is "$$ - $$$"
# when fill all nan  with most popular

food_data["Price Range"].fillna(
    food_data["Price Range"].mode()[0], inplace=True
)
# check values
print(food_data["Price Range"].value_counts())
# Convert "Price Range" to numeerical
price_range_dict = {"$": 1.0, "$$ - $$$": 2.0, "$$$$": 3.0}
food_data["Price Range"] = food_data["Price Range"].apply(
    lambda x: price_range_dict[x]
)
# check
print(food_data["Price Range"].value_counts())
print(food_data.info())
food_data["City"].value_counts()
# use only capitals in dataset
world_cityes = world_cityes.dropna(subset=["capital"])

# change Oporto to Porto - it's same
food_data.loc[food_data["City"] == "Oporto", "City"] = "Porto"


def get_dict_by_city(param):
    return pd.Series(
        world_cityes[param].values, index=world_cityes["city_ascii"]
    ).to_dict()


population_dict = get_dict_by_city("population")
capital_dict = get_dict_by_city("capital")
country_code_dict = get_dict_by_city("iso3")

food_data["population in thousands"] = food_data["City"].apply(
    lambda x: population_dict[x] / 1000
)
food_data["capital"] = food_data["City"].apply(lambda x: capital_dict[x])
food_data["country_code"] = food_data["City"].apply(
    lambda x: country_code_dict[x]
)

display(food_data.info())
# country_pop_dict = pd.Series(
#     country_population["2018"].values, index=country_population["Country Code"]
# ).to_dict()
# # country_pop_dict = country_population[['Country Code','2018']].to_dict()


# food_data["Population in country"] = food_data["country_code"].apply(lambda x: country_pop_dict[x])

fig = go.Figure(
    data=[
        go.Bar(
            name="Rest peer City",
            x=food_data["City"].value_counts().index,
            y=food_data["City"].value_counts(),
        ),
        go.Bar(
            name="Population",
            y=food_data["population in thousands"].value_counts().index,
            x=food_data["City"].value_counts().index
        ),
    ]
)
# Change the bar mode
fig.update_layout(barmode="group")
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Rest peer City",
        "y": 0.85,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="City name",
    yaxis_title="Rest count",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
# add new vars: 1. rest count per City, 2. rest count per 1 000 population
rest_counts_dict = food_data["City"].value_counts().to_dict()
food_data["Rest counts per City"] = food_data["City"].apply(
    lambda x: rest_counts_dict[x]
)
food_data["Rest counts per 1000"] = (
    food_data["Rest counts per City"] / food_data["population in thousands"]
)
display(food_data.info())
fig = go.Figure(
    data=[
        go.Bar(
            name="Rest peer Country",
            x=food_data["country_code"].value_counts().index,
            y=food_data["country_code"].value_counts(),
        )
    ]
)
# Change the bar mode
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Rest peer Country",
        "y": 0.85,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Country code",
    yaxis_title="Rest count",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
# add new var - rest count per Country
rest_counts_dict = food_data["country_code"].value_counts().to_dict()
food_data["Rest counts per Country"] = food_data["country_code"].apply(
    lambda x: rest_counts_dict[x]
)
display(food_data["Rest counts per Country"].value_counts())
# replace NaN to ['Other'] to prevent assertion of ast.literal_eval
food_data["Cuisine Style"].fillna("['Other']", inplace=True)
food_data["Cuisine Style"] = food_data["Cuisine Style"].apply(
    lambda x: ast.literal_eval(x)
)

flat_list = [
    item for sublist in food_data["Cuisine Style"] for item in sublist
]
cuisine_style_counter = Counter(flat_list)

print(len(cuisine_style_counter))
pp.pprint((take(10, cuisine_style_counter.most_common())))
# Cuisines to dummyes
for cuisine_style in cuisine_style_counter:
    food_data[cuisine_style] = food_data["Cuisine Style"].apply(
        lambda x: 1 if cuisine_style in x else 0
    )
food_data["Cuisine Style Count"] = food_data["Cuisine Style"].apply(
    lambda x: len(x)
)
# get mean cousine styles per restaurant
pp.pprint(food_data["Cuisine Style Count"].mean())
display(food_data.head(5))
fig = px.box(food_data, x="Cuisine Style Count", orientation='h')
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "BoxPlot по количеству кухонь",
        "y": 0.95,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Количество кухонь",    
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
# food_data["Reviews"] = food_data["Reviews"].str.lower()
food_data["Number of Reviews"].mean()


food_data["Reviews"] = food_data["Reviews"].str.replace("nan", "' '")
# remove 'nan' for prevent catboost fall with assertion
food_data["Reviews"].fillna("[[], []]", inplace=True)

# for debug only(for catch assertion) instead use lambda x: tokenizer.tokenize(x)
# def tokenize_review_debug(tok):
#    try:
#      ret = tokenizer.tokenize(tok)
#    except:
#      print(tok)
#    return ret


# for debug only(for catch assertion) instead use lambda x: ast.literal_eval(x)
# def ast_review_debug(tok):
#   try:
#     ret = ast.literal_eval(tok)
#   except:
#         print(tok)
#     return ret


food_data["Reviews"] = food_data["Reviews"].apply(
    lambda x: ast.literal_eval(x)
)

# how many reviews have the date and text?
# first of all get max and min list of dates len
reviews_dates_list_len = food_data["Reviews"].apply(lambda x: len(x[1]))

print(f"Maximum Reviews with text`s: {reviews_dates_list_len.max()}")
print(f"Minimum Reviews with text`s: {reviews_dates_list_len.min()}")

# ok min is 0 it's mean that we have Reviews with no date
# check how many it is?
reviews_dates_list_0 = food_data["Reviews"].apply(
    lambda x: 1 if len(x[1]) == 0 else 0
)
print(f"Number of rows with no date in Review: {reviews_dates_list_0.sum()}")

# now check how many records have not Review at all
reviews_list_0 = food_data["Reviews"].apply(
    lambda x: 1 if len(x[0]) == 0 else 0
)
print(f"Number of rows with no text in Review:{reviews_dates_list_0.sum()}")

# hmm these numbers are equal, bith is 8114
# now check what we have not rows with dates in Review but without Reviews text
# and save this in dataset

food_data["No Dates or No texts in Reviews"] = food_data["Reviews"].apply(
    lambda x: 1
    if (len(x[0]) == 0 and len(x[1]) != 0)
    or (len(x[1]) == 0 and len(x[0]) != 0)
    else 0
)

print(
    f"Number of rows with text but no date and vice versa: \
    {food_data['No Dates or No texts in Reviews'].sum()}"
)
# convert text to date
food_data["Reviews Dates"] = food_data["Reviews"].apply(
    lambda x: [datetime.strptime(d, "%m/%d/%Y") for d in x[1]]
)

# Check if dates in list in right order and reordering if order is wrong
food_data["Reviews Dates"] = food_data["Reviews Dates"].apply(
    lambda x: [x[i] for i in [1, 0]] if ((len(x) > 1) and (x[1] > x[0])) else x
)

# Add diffs between first and last review to dataset
food_data["Reviews Dates Diff"] = food_data["Reviews Dates"].apply(
    lambda x: (x[0] - x[1]).days if len(x) > 1 else 0
)

# Add number of reviews per restaurant to dataset
food_data["Number of Reviews per rest"] = food_data["Reviews Dates"].apply(
    lambda x: len(x)
)


print(
    f"Maximal number of days between Reviews: \
    {food_data['Reviews Dates Diff'].max()}"
)

print(
    f"Minimal number of days between Reviews: \
    {food_data['Reviews Dates Diff'].min()}"
)


print(
    f"Mean number of days between Reviews: \
    {food_data['Reviews Dates Diff'].mean()}"
)

dates_list = food_data["Reviews Dates"].to_list()
dates_flat = [item for sublist in dates_list for item in sublist]
latest_review = sorted(dates_flat, reverse=True)[0]
earlest_review = sorted(dates_flat)[0]
print(f"Earlest review date: {earlest_review}")
print(f"Latest review date: {latest_review}")

print(
    f'Reviews per restaraunt value counts:\n {food_data["Number of Reviews"].value_counts().to_frame()}'
)

# Add information how far current review is from latest review
food_data["How far from latest review"] = food_data["Reviews Dates"].apply(
    lambda x: (latest_review - x[0]).days
    if len(x) > 0
    else (latest_review - earlest_review).days
)
# add new var - number of Reviews per City
review_counts_dict = (
    food_data.groupby(["City"])["Number of Reviews"].sum().to_dict()
)

food_data["Number of Reviews per City"] = food_data["City"].apply(
    lambda x: review_counts_dict[x]
)

# add new var - number of Reviews per Country
review_counts_dict = (
    food_data.groupby(["country_code"])["Number of Reviews"].sum().to_dict()
)

food_data["Number of Reviews per Country"] = food_data["country_code"].apply(
    lambda x: review_counts_dict[x]
)
food_data.info()
display(food_data.head(5))
# lets go to work with Review texts

# add  "Number" to token_types for number processing
tokenizer = Tokenizer(
    lowercasing=True, separator_type="BySense", token_types=["Word"]
)


stop_words = set(
    ("be", "is", "are", "the", "an", "of", "and", "in", "food", "a")
)


def filter_stop_words(tokens):
    return list(filter(lambda x: x not in stop_words, tokens))


reviews_tokens = food_data["Reviews"].apply(
    lambda x: [
        item
        for sublist in [tokenizer.tokenize(i) for i in x[0]]
        for item in sublist
    ]
)
reviews_tokens = [filter_stop_words(tokens) for tokens in reviews_tokens]

# convert list of lists to list
flat_list = [item for sublist in reviews_tokens for item in sublist]
tokens_counter = Counter(flat_list)

all_tokens_count = 0

review_meaning_words = [
    "awesome",
    "excellent",
    "great",
    "good",
    "ok",
    "bad",
    "awful",
    "terrible",
    "horrible",
]


for w in review_meaning_words:
    try:
        all_tokens_count += tokens_counter[w]
        print(f'Token {w} occured {tokens_counter[w]} times')
    except KeyError:
        print(f"word {w} not found!")

print(f"Total tokens count: {all_tokens_count}")
# import nltk
# import os

# nltk_data_path = os.path.join(os.path.dirname(nltk.__file__), 'nltk_data')
# nltk.data.path.append(nltk_data_path)
# nltk.download('wordnet', nltk_data_path)

# lemmatizer = nltk.stem.WordNetLemmatizer()

# def lemmatize_tokens_nltk(tokens):
#     return list(map(lambda t: lemmatizer.lemmatize(t), tokens))


# text_small_lemmatized_nltk = [lemmatize_tokens_nltk(tokens) for tokens in reviews_tokens]
# # convert list of lists to list
# flat_list = [item for sublist in text_small_lemmatized_nltk for item in sublist]
# text_small_lemmatized_nltk_counter = Counter(flat_list)

# total_lemmas_count = 0
# for w in [
#     "awesome",
#     "excellent",
#     "great",
#     "good",
#     "ok",
#     "bad",
#     "awful",
#     "terrible",
#     "horrible",
# ]:
#     try:
#         # print(text_small_lemmatized_nltk_counter[w])
#         total_lemmas_count += text_small_lemmatized_nltk_counter[w]
#     except:
#         print(f"word {w} not found!")

# print(f'Total lemmas count: {total_lemmas_count}')
# print(f'Lemmas and tokens count diff: {total_lemmas_count-all_tokens_count}')
def review_word_counter(rev, word):
    ret = 0
    for i in rev[0]:
        ret += i.count(word)
    return ret


for w in review_meaning_words:
    food_data[w] = food_data["Reviews"].apply(review_word_counter, args=(w,))
display(food_data.head(5))
# remove first 'd' character from ID_TA parameter and convert it to digit
food_data["ID_TA"] = food_data["ID_TA"].apply(lambda x: int(x[1:]))
# read group ID from URL

food_data["Group ID from URL_TA"] = food_data["URL_TA"].str.extract(
    pat="(-g\d+)"
)
food_data["Group ID from URL_TA"] = food_data["Group ID from URL_TA"].apply(
    lambda x: x[1:]
)
display(food_data["Group ID from URL_TA"].value_counts())
url_ta_group_counter = Counter(food_data["Group ID from URL_TA"].to_list())

print(len(url_ta_group_counter))
pp.pprint((take(10, url_ta_group_counter.most_common())))
# looks like it's some geografic region, may be it's city suburb
# rest counts per "Group ID from URL_TA"

# rest_counts_dict = food_data["Group ID from URL_TA"].value_counts().to_dict()
# food_data["Rest counts peer Group ID from URL_TA"] = food_data[
#     "Group ID from URL_TA"
# ].apply(lambda x: rest_counts_dict[x])
# add new var - number of Reviews per Group ID from URL_TA
# review_counts_dict = (
#     food_data.groupby(["Group ID from URL_TA"])["Number of Reviews"].sum().to_dict()
# )

# food_data["Number of Reviews per Group ID from URL_TA"] = food_data["Group ID from URL_TA"].apply(
#     lambda x: review_counts_dict[x]
# )

# URL_TA grpups to dummyes
for url_ta_group in url_ta_group_counter:
    food_data[url_ta_group] = food_data["Group ID from URL_TA"].apply(
        lambda x: 1 if url_ta_group in x else 0
    )
fig = px.box(food_data, x="Ranking", orientation='h')
fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Ranking BoxPlot",
        "y": 0.95,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Rank",    
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)
fig.show()
fig = px.histogram(
    food_data[
        food_data["City"].isin(
            ["London", "Paris", "Madrid", "Barcelona", "Berlin"]
        )
    ],
    x="Ranking",
    nbins=200,
    color="City",
)

fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Ranking Distribution",
        "y": 0.92,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Rank",
    yaxis_title="Rest count",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)

fig.show()
fig = px.histogram(
    food_data[
        food_data["City"].isin(
            ["Heelsinki", "Bratislava", "Luxembourg", "Ljubljana", "Oslo"]
        )
    ],
    x="Ranking",
    nbins=200,
    color="City",
)

fig.update_layout(
    {"plot_bgcolor": "rgba(0, 0, 0, 0)", "paper_bgcolor": "rgba(0, 0, 0, 0)",},
    title={
        "text": "Ranking Distribution",
        "y": 0.92,
        "x": 0.4,
        "xanchor": "center",
        "yanchor": "top",
    },
    xaxis_title="Rank",
    yaxis_title="Rest count",
    font={"family": "Courier New, monospace", "size": 14, "color": "#7f7f7f"},
)

fig.show()
# add new var - Ranking per Country
review_counts_dict = (
    food_data.groupby(["country_code"])["Ranking"].sum().to_dict()
)

food_data["Ranking per Country"] = food_data["country_code"].apply(
    lambda x: review_counts_dict[x]
)


food_data["Ranking per Country"] = (
    food_data["Ranking per Country"] / food_data["Rest counts per Country"]
)
# Create four new params

food_data["Relative Ranking per Rest counts (City)"] = (
    food_data["Ranking"] / food_data["Rest counts per City"]
)

food_data["Relative Ranking per Number of Reviews (City)"] = (
    food_data["Ranking"] / food_data["Number of Reviews per City"]
)

food_data["Relative Ranking per Rest counts (Country)"] = (
    food_data["Ranking per Country"] / food_data["Rest counts per Country"]
)

food_data["Relative Ranking per Number of Reviews (Country)"] = (
    food_data["Ranking per Country"] / food_data["Number of Reviews per Country"]
)


food_data = pd.get_dummies(
    food_data, columns=["country_code", "capital"]
)
food_data = food_data.select_dtypes(exclude=['object'])
food_data.info()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


food_data_train = food_data[food_data["sample"] == 1].drop(["sample"], axis=1)
food_data_test = food_data[food_data["sample"] == 0].drop(["sample"], axis=1)

y = food_data_train["Rating"].values
X = food_data_train.drop(["Rating"], axis=1)

RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
model = RandomForestRegressor(
    n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred * 2) / 2

print("MAE:", metrics.mean_absolute_error(y_test, y_pred))

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
display(feat_importances.nlargest(15))
test_data = food_data_test.drop(["Rating"], axis=1)
sample_submission = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/sample_submission.csv')
test_data
predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission*2)/2
predict_submission.shape
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)
import pandas as pd

import numpy as np
movies = pd.read_csv("../input/pandas-cookbook-data/data/movie.csv")

movies.head()
cols = [

    "actor_1_name",

    "actor_2_name",

    "actor_3_name",

    "director_name"

]

movies_actor_director = movies[cols]

movies_actor_director.head()
type(movies[["director_name"]]) # if i pass a list on the index operation it will return a DataFrame...
type(movies["director_name"]) # if i pass just a string on the index operation it will return a Series...
print(type(movies.loc[:, ["director_name"]]))

movies.loc[:, ["director_name"]].head()
print(type(movies.loc[:, "director_name"]))

movies.loc[:, "director_name"].head()
def shorten(col):

    #print("col type:", type(col), "col:", col)

    return(

        str(col)

        .replace("facebook_likes", "fb")

        .replace("_for_reviews", "")

    )



movies = movies.rename(columns=shorten)

movies.head()
movies.dtypes.value_counts()
movies.select_dtypes(include="int").head()
movies.select_dtypes(include="number").head()
movies.select_dtypes(include=["int", "object"]).head(3)
movies.select_dtypes(exclude="float").head(3)
movies.filter(like="fb").head(3)
# the 'cols' list is defined above...

movies.filter(items=cols).head(3)
movies.filter(regex=r"\d").head(3) ## searching for a column that have a digit somewhere in their name.
movies.columns
## cat: categorical

## cont: continuous

cat_core = [ #1

    "movie_title",

    "title_year",

    "content_rating",

    "genres"

]

cat_people = [ #2

    "director_name",

    "actor_1_name",

    "actor_2_name",

    "actor_3_name"

]

cat_other = [ #3

    "color",

    "country",

    "language",

    "plot_keywords",

    "movie_imdb_link"

]

cont_fb = [ #4

    "director_fb",

    "actor_1_fb",

    "actor_2_fb",

    "actor_3_fb",

    "cast_total_fb",

    "movie_fb"

]

cont_finance = [ #5

    "budget",

    "gross"

]

cont_num_reviews = [ #6

    "num_voted_users",

    "num_user",

    "num_critic"

]

cont_other = [ #7

    "imdb_score",

    "duration",

    "aspect_ratio",

    "facenumber_in_poster"

]
new_col_order = (

    cat_core

    + cat_people

    + cat_other

    + cont_fb

    + cont_finance

    + cont_num_reviews

    + cont_other

)
set(movies.columns) == set(new_col_order)
movies = movies[new_col_order]

movies.head(3)
movies.shape
movies.size
movies.ndim
len(movies) # when a DataFrame is passed to the built-in len() function it returns the number of rows...
movies.count().head()
movies.select_dtypes(include="number").min() # I could also use .max() .mean() .median() .std()
movies.select_dtypes(include="number").min(skipna=False).head() ## with skipna=False, only numeric columns with missing values will calculate a result.
movies.describe().T.head(3)
movies.describe(percentiles=[0.01, 0.3, 0.99]).T.head(3) ## using the 'percentiles' paramenter
movies.isnull().head()
movies.isna().head()
movies.isnull().sum().head(10) # counts the number of missing values in each column
movies.isnull().sum().sum()
movies.isnull().any().any()
movies.isnull().dtypes.value_counts()
(

    movies

    .select_dtypes(["object"])

    .columns

    .to_list()

)
colleges = pd.read_csv("../input/pandas-cookbook-data/data/college.csv", index_col="INSTNM")
colleges.shape
colleges.head(3)
try:

    colleges + 5

    print("it was possible to do that operation...")

except TypeError:

    print("it is only possible to do this operation if the DataFrame has an homogeneous numeric data type...")
college_ugds = colleges.filter(like="UGDS_")

print(college_ugds.shape)

college_ugds.head()
try:

    college_ugds_plus_5 = college_ugds + 5

    print("it was possible to do that operation...")

except TypeError:

    print("it is only possible to do this operation if the DataFrame has an homogeneous numeric data type...")
college_ugds_plus_5.head()
name = "Northwest-Shoals Community College"

college_ugds.loc[name]
college_ugds.loc[name].round(2)
(college_ugds.loc[name] + 0.0001).round(2)
(college_ugds + 0.00501).head()
((college_ugds + 0.00501) // 0.01).head() # rounding to the nearest whole number percentage
college_ugds_op_round = ((college_ugds + 0.00501) // 0.01 / 100)

college_ugds_op_round.head()
college_ugds_round = (college_ugds + 0.00001).round(2)

college_ugds_round.head()
college_ugds_op_round.equals(college_ugds_round)
0.045 + 0.005
college2 = (

    college_ugds

    .add(0.00501)

    .floordiv(0.01)

    .div(100)

)

college2.head()
college2.equals(college_ugds_op_round)
np.nan
np.nan == np.nan
None == None
print(np.nan > 5)

print(5 >= np.nan)

print(np.nan != 5)
(college_ugds == 0.0019).head(3)
(college_ugds == 0.0019).sum().sum()
college_self_compare = college_ugds == college_ugds

college_self_compare.head()
college_self_compare.all() # the == operator does not work well with the missing values
(college_ugds == np.nan).sum()
college_ugds.isna().sum()
college_ugds.equals(college_ugds)
college_ugds.eq(0.0019).head(3) # same as college_ugds == .0019
from pandas.testing import assert_frame_equal
assert_frame_equal(college_ugds, college_ugds) is None
college_ugds.count() # by default axis=0 or "index", index direction: vertical |
college_ugds.count(axis="columns") # columns is the direction, horizontal ------
college_ugds.sum(axis="columns").head()
college_ugds.median()#axis="index")
college_ugds_cumsum = college_ugds.cumsum(axis=1)

college_ugds_cumsum.head()
pd.read_csv("../input/pandas-cookbook-data/data/college_diversity.csv", index_col="School")
(

    college_ugds

    .isnull()

    .sum(axis="columns")

    .sort_values(ascending=False)

)#.sum()
college_ugds.shape
college_ugds = college_ugds.dropna(how="all")

college_ugds.shape
college_ugds.isnull().sum()
college_ugds.ge(0.15).head()
diversity_metric = college_ugds.ge(0.15).sum(axis="columns")

diversity_metric.sort_values(ascending=False).head()
diversity_metric.value_counts()
college_ugds.loc[

    [

        "Regency Beauty Institute-Austin",

        "Central Texas Beauty College-Temple"

    ]

]
us_news_top = [

    "Rutgers University-Newark",

    "Andrews University",

    "Stanford University",

    "University of Houston",

    "University of Nevada-Las Vegas"

]

diversity_metric.loc[us_news_top]
(college_ugds > 0.01).all(axis=1).any()
print("tank you!")
import pandas as pd
from learntools.advanced_pandas.renaming_and_combining import *
pd.set_option('max_rows', 5)
reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
check_q1(pd.DataFrame())
reviews.head()
reviews.rename(columns={'region_1':'region','region_2':'locale'},inplace=True)

reviews.set_index(reviews.winery)

gaming_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("../input/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"
import pandas as pd
product = pd.concat([gaming_products,movie_products],ignore_index=True)
product
powerlifting_meets = pd.read_csv("../input/powerlifting-database/meets.csv")
powerlifting_competitors = pd.read_csv("../input/powerlifting-database/openpowerlifting.csv")
import pandas as pd
powerlifting_meets.merge(powerlifting_competitors,on='MeetID')

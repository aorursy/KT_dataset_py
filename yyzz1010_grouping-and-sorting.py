import pandas as pd



reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

#pd.set_option("display.max_rows", 5)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.grouping_and_sorting import *

print("Setup complete.")
# Your code here

reviews_written = reviews['taster_twitter_handle'].value_counts()



q1.check()
q1.hint()

q1.solution()
best_rating_per_price = reviews.groupby('price').max().sort_index()



q2.check()
q2.hint()

q2.solution()
price_extremes = pd.DataFrame([reviews.groupby('variety')['price'].min(), reviews.groupby('variety')['price'].max()],[reviews.variety])



q3.check()
q3.hint()

q3.solution()
sorted_varieties = ____



q4.check()
q4.hint()

q4.solution()
reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()



q5.check()
#q5.hint()

#q5.solution()
reviewer_mean_ratings.describe()
country_variety_counts = reviews.groupby(['country', 'variety']).sort_values(descending=True)



q6.check()
q6.hint()

q6.solution()
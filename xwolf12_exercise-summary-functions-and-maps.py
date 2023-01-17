import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
reviews.columns
median_points = reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.value_counts().index.values

countries
print(reviews.country.value_counts().index.values)

print()

# look that the exercise counts NaN as a value

print(reviews.country.unique())
countries = reviews.country.unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
reviews_price_mean = reviews.price.mean()

centered_price = reviews.price.map(lambda x: x - reviews_price_mean)



q4.check()
#q4.hint()

#q4.solution()
bargain_idx = (reviews.points / reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx,'title']

bargain_wine
bargain_wine = bargain_wine



q5.check()
# q5.hint()

# q5.solution()
tropicals = reviews.description.apply(lambda x: "tropical" in x ).sum()

fruites =  reviews.description.apply(lambda x: "fruity" in x ).sum()

descriptor_counts = pd.Series([tropicals, fruites], index=["tropical", "fruity"])



# another way but slower than the first one

# tropicals_an = 0

# fruites_an = 0



# for index,row  in reviews.iterrows():

#     if "tropical" in row["description"] :

#         tropicals_an+=1

#     if "fruity" in row["description"]:

#         fruites_an+=1



# descriptor_counts = pd.Series([tropicals_an, fruites_an], index=["tropical", "fruity"])



q6.check()
# q6.hint()

# q6.solution()
star_ratings = pd.np.where(3, 2, reviews.points >=95)

star_ratings = pd.Series(pd.np.where(1, star_ratings, reviews.points < 85))



q7.check()
#q7.hint()

#q7.solution()
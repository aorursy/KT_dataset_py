import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



# Check your answer

q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



# Check your answer

q2.check()
#q2.hint()

#q2.solution()
reviews.country.value_counts()
reviews_per_country =reviews.country.value_counts()



# Check your answer

q3.check()
q3.hint()

#q3.solution()
centered_price = reviews.price.map(lambda p:p-reviews.price.mean())



# Check your answer

q4.check()
#q4.hint()

#q4.solution()
bargain_idx=(reviews.points/reviews.price).idxmax()

bargain_wine = reviews.loc[bargain_idx,'title']

# bargain_idx

# Check your answer

q5.check()

# bargain_wine
q5.hint()

q5.solution()
# n_tropical=reviews.description.str.count("tropical").sum()

# n_fruity = reviews.description.str.count("fruity").sum()

n_tropical = reviews.description.map(lambda desc: "tropical" in desc).sum()

n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()

descriptor_counts = pd.Series([n_tropical,n_fruity],index=["tropical","fruity"])

# descriptor_counts

# n_tropical[1000:1005]

# Check your answer

q6.check()
#q6.hint()

q6.solution()
reviews.country.value_counts()['Canada']
def stars(row):

    if row.country=='Canada':

        return 3

    elif row.points>=95:

        return 3

    elif row.points>=85:

        return 2

    else:

        return 1

star_ratings = reviews.apply(stars,axis=1)



# Check your answer

q7.check()
#q7.hint()

q7.solution()
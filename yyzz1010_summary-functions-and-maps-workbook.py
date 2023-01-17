import pandas as pd

pd.set_option("display.max_rows", 5)

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)



from learntools.core import binder; binder.bind(globals())

from learntools.pandas.summary_functions_and_maps import *

print("Setup complete.")



reviews.head()
median_points = reviews.points.median()



q1.check()
#q1.hint()

#q1.solution()
countries = reviews.country.unique()



q2.check()
#q2.hint()

#q2.solution()
reviews_per_country = reviews.country.value_counts()



q3.check()
#q3.hint()

#q3.solution()
centered_price = reviews.price.mean()



q4.check()
q4.hint()

q4.solution()
reviews['points-to-price_ratio'] = reviews.points / reviews.price

index = reviews.idxmax(['points-to-price_ratio'])

bargain_wine = reviews.title.loc[index]



q5.check()
q5.hint()

q5.solution()
f1 = reviews.description.isin(['tropical'])

f2 = reviews.description.isin(['fruity'])



descriptor_counts = pd.Series([f1.value_counts(),f2.value_counts], ['tropical', 'fruity'])



q6.check()
descriptor_counts



q6.hint()

q6.solution()
def star(reviews):

    if reviews.country == 'Canada':

        reviews.star == 3

        

    elif reviews.points >= 95:

        reviews.star == 3

        

    elif reviews.points >=85: 

        reviews.star == 2

            

    else: 

        reviews.star == 1



    return reviews



star_ratings = reviews.apply(star, axis = 'columns')



q7.check()
q7.hint()

q7.solution()
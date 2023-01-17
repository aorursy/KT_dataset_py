import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For find dataset location
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
from langdetect import detect
# Take dateset
reviews_df = pd.read_csv('../input/reviews-of-londonbased-hotels/London_hotel_reviews.csv',  encoding = "ISO-8859-1" )

# Look, how many reviews in the orginal dataset
totalReviews = reviews_df.shape
print(totalReviews)        # There are 27330 reviews and 1 columns
reviews_df.tail()          # Show end of the 5 reviews

# Remove some reviews created full of unknown characters.
reviews_df = reviews_df[reviews_df['Review Text'].str.contains("<U") == False] 

# The total number of reviews after removed reviews are full of unknown character
print(reviews_df.shape) # There are 26877 reviews.


otherLang = []
j = 1
for t in range(len(reviews_df)):                  # iterate for each object
    i = str(reviews_df['Review Text'].values[t])  # Take just reviews to String : i
    if detect(i) != "en":                         # detect(review) is English? if it is not do:
        otherLang.append(i)
#         print(j, i)
        j += 1

for i in range(10):
    print(i, otherLang[i], '\n' )

print("\nTotal reviews: {} \n Created by unknown characters: {} \n Other Languages reviews: {} \n English Reviews: {}".format(totalReviews[0], totalReviews[0]-len(reviews_df), j, len(reviews_df) - j))

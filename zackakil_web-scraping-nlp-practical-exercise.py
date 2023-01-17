import sys
if sys.version_info[0] == 3:
    print('Great! Python 3! Lets get on with scraping!')
else:
    print('Yikes! Python 2! This may not work for you! ')
# 'requests' is what we use to send web-requests (to fetch the html files from websites)
import requests

# beautiful-soup will help us in navigating through the html extract just the text we care about
from bs4 import BeautifulSoup
WEB_PAGE_TO_SCRAPE_URL = "https://techcrunch.com/"
# send request for the web page
# WRITE YOUR CODE HERE


# lets look at some of the raw text (the html), more specificly the first 500 characters 
# WRITE YOUR CODE HERE



# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


# YOUR CODE GOES HERE


element_search = None

# YOUR CODE STARTS HERE




# YOUR CODE ENDS HERE

# I won't print it all becuase it's quite long
# let's just see if all of the attributes match

if element_search:
    element_search.attrs
article_listings = None

# YOUR CODE STARTS HERE




# YOUR CODE ENDS HERE

if article_listings:

    print('Numer of article:', len(article_listings))
    print('Printing article titles \n')

    # we know most of these elements has an attribute called 'data-sharetitle' that stores the articles title
    # so lets print these out

    for a in article_listings:
        if 'data-sharetitle' in a.attrs:
            print(a['data-sharetitle'])
# YOUR CODE GOES HERE (MAKE AS MANY NEW CELLS AS YOU LIKE)



# YOUR CODE GOES HERE



# YOUR CODE GOES HERE



# textblob has a pre-trained sentiment analysis model that we can use
from textblob import TextBlob
# YOUR CODE GOES HERE


# WRITE YOUR CODE HERE OR COPY AND PASTED FROM ABOVE



# import ploting library 
import matplotlib.pyplot as plt

# draw plot in notebook
%matplotlib inline
# COPY AND PAST THE CODE FROM ABOVE 





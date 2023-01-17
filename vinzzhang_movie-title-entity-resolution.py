import pandas as pd
import numpy as np
import difflib
#Determine the average rating for each movie, compensating for the bad spelling in the ratings file.
mlist = open("../input/movie-list/movie_list.txt").readlines() 
df = pd.DataFrame(mlist,columns=['movie'])
df['movie'] = df['movie'].str.replace('\n','')
rate = pd.read_csv("../input/movie-ratings/movie_ratings.csv")
rate['title'] = rate['title'].apply(lambda x: difflib.get_close_matches(x,df['movie']))
rate['title'] = rate['title'].apply(lambda x: tuple(x))  #avoid unhashable list
mean = rate.groupby('title')['rating'].mean().round(2)
output = pd.DataFrame(mean, rate['title'], columns = ['rating'])
output = output.drop_duplicates().sort_index().iloc[1:]
output



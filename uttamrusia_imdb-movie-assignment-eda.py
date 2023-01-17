# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import the numpy and pandas packages

import numpy as np
import pandas as pd
movies = movies = pd.read_csv("../input/imdb-movie-data/movie_IMDB.csv")# Write your code for importing the csv file here
movies.head()
# Write your code for inspection here
print(movies.shape)
print(movies.info())
# Write your code for column-wise null count here
print(movies.isnull().sum())
# Write your code for row-wise null count here
movies.isnull().sum(axis=1)
# Write your code for column-wise null percentages here
round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 
movies = movies.drop('color', axis=1)
movies = movies.drop('director_facebook_likes', axis=1)
movies = movies.drop('actor_1_facebook_likes', axis=1)
movies = movies.drop('actor_2_facebook_likes', axis=1)
movies = movies.drop('actor_3_facebook_likes', axis=1)
movies = movies.drop('actor_2_name', axis=1)
movies = movies.drop('cast_total_facebook_likes', axis=1)
movies = movies.drop('actor_3_name', axis=1)
movies = movies.drop('duration', axis=1)
movies = movies.drop('facenumber_in_poster', axis=1)
movies = movies.drop('content_rating', axis=1)
movies = movies.drop('country', axis=1)
movies = movies.drop('movie_imdb_link', axis=1)
movies = movies.drop('aspect_ratio', axis=1)
movies = movies.drop('plot_keywords', axis=1)


round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for dropping the rows here
movies = movies[~np.isnan(movies['gross'])]
movies = movies[~np.isnan(movies['budget'])]

round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for filling the NaN values in the 'language' column here
movies["language"].fillna("English", inplace = True)
round(100*(movies.isnull().sum()/len(movies.index)), 2)
# Write your code for checking number of retained rows here
print(movies.shape)
# Write your code for unit conversion here
movies['budget'] = movies['budget'].apply(lambda x: x/1000000)
movies['gross'] = movies['gross'].apply(lambda x: x/1000000)
# Write your code for creating the profit column here
movies['profit'] = movies['gross'] - movies['budget']
# Write your code for sorting the dataframe here
movies = movies.sort_values('profit',ascending=False)
# Write code for profit vs budget plot here
movies.plot.scatter(x="budget",y="profit")

#yes outliers are present in data
# Write your code to get the top 10 profiting movies here
top10 = movies.head(10)
# Write your code for dropping duplicate values here
top10.drop_duplicates(inplace = True)
# Write code for repeating subtask 2 here
#Create a new column called profit which contains the difference of the two columns: gross and budget.// Done
#Sort the dataframe using the profit column as reference. // Done
#Plot profit (y-axis) vs budget (x- axis) and observe the outliers using the appropriate chart type.// Below
#Extract the top ten profiting movies in descending order and store them in a new dataframe - top10 // Done
top10.plot.scatter(x="budget",y="profit")
# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'
IMDb_Top_250 = movies [(movies['num_voted_users']>25000)]
IMDb_Top_250 = (IMDb_Top_250.sort_values(by=['imdb_score'],ascending = False)).head(250)
IMDb_Top_250['Rank'] = np.arange(1,251)
Top_Foreign_Lang_Film = IMDb_Top_250 [(IMDb_Top_250['language'] != 'English')] 
# Write your code to extract top foreign language films from 'IMDb_Top_250' here
# Write your code for extracting the top 10 directors here
top10director = movies.pivot_table(values = 'imdb_score',index = 'director_name',aggfunc = 'mean')
top10director = (top10director.sort_values(by=['imdb_score'],ascending = False)).head(10)
top10director = (top10director.sort_values(by=['director_name'],ascending = True)).head(10)
print(top10director.head(10))
# Write your code for extracting the first two genres of each movie here
genre_list = movies['genres'].str.split ("|", expand =True)
movies ['genre_1'] = genre_list[0]
movies ['genre_2'] = genre_list[1]
movies['genre_2'].fillna(movies['genre_1'], inplace = True)
movies_by_segment =movies.groupby(['genre_1','genre_2'])
# Write your code for grouping the dataframe here
PopGenre = movies_by_segment['gross'].mean()
PopGenre = PopGenre.sort_values(ascending=False)
print(PopGenre)
# Write your code for getting the 5 most popular combo of genres here
# Write your code for creating three new dataframes here
Meryl_Streep = movies [(movies['actor_1_name'] == 'Meryl Streep')] 
# Include all movies in which Meryl_Streep is the lead
Leo_Caprio = movies [(movies['actor_1_name'] == 'Leonardo DiCaprio')]
# Include all movies in which Leo_Caprio is the lead
Brad_Pitt = movies [(movies['actor_1_name'] == 'Brad Pitt')]
# Include all movies in which Brad_Pitt is the lead
# Write your code for combining the three dataframes here
combined_df = pd.concat([Meryl_Streep, Leo_Caprio ,Brad_Pitt], axis=0)
#Meryl_Streep, Leo_Caprio, and Brad_Pitt
# Write your code for grouping the combined dataframe here
group_df = combined_df.groupby(['actor_1_name'])
# Write the code for finding the mean of critic reviews and audience reviews here
group_df['num_user_for_reviews','num_critic_for_reviews'].mean()
# Write the code for calculating decade here
movies['decade'] = movies['title_year'].apply(lambda x: (x//10)*10)
# Write your code for creating the data frame df_by_decade here 
df_by_decade=movies.pivot_table(index='decade',values = 'num_voted_users',aggfunc='sum')
df_by_decade= df_by_decade.reset_index()
# Write your code for plotting number of voted users vs decade
import matplotlib.pyplot as plt

plt.plot(df_by_decade['decade'],df_by_decade['num_voted_users'] )

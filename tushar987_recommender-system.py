# Import Packages



from math import sqrt

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



# Getting more than one output Line

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# Getting the Dataset



movies= pd.read_csv("../input/movies.csv") # ,nrows=5000)

movies.head()



ratings=pd.read_csv("../input/ratings.csv",usecols=['userId','movieId','rating']) #,nrows=1000)

ratings.head()



tags= pd.read_csv('../input/tags.csv')

tags.head()



# Deleting unnecessary columns

## for time being we will not consider the timestamp of movie for our analysis





del tags['timestamp']
movies.info()
ratings['rating'].describe(include='all')
ratings.groupby('rating')['movieId'].nunique()

%matplotlib inline

ratings.hist(column='rating',figsize=(10,10),bins=5,grid=False)
tag_counts = tags['tag'].value_counts()

tag_counts[:15]

tag_counts[:15].plot(kind='bar' , figsize=(10,10))
movies['movieId'].count()

genre_filter= (movies['genres'] == '(no genres listed)')

## Removing movies with no genre



movies=movies[~genre_filter]

movies=movies.reset_index(drop=True)  ## God bless StackOverflow.. Remeber removing filtered rows does not reindex the dataframe



genres_count= {}

for row in range(movies['movieId'].count()):

    for genre in movies['genres'][row].split("|"):

        if(genre != ''):

            genres_count[genre]= genres_count.get(genre,0)+1

        

genres_count

fig, ax = plt.subplots(figsize=(15,10))

plt.barh(range(len(genres_count)), list(genres_count.values()))

plt.yticks(range(len(genres_count)),list(genres_count.keys()))

plt.xlabel('Movie Count')

plt.title("Genre Popularty")

for i, v in enumerate(genres_count.values()):

    ax.text(v + 20, i + .10, v, color='blue', fontweight='bold')
def euclidean_distance(person1,person2):

    #Getting details of person1 and person2

    df_first= ratings.loc[ratings['userId']==person1]

    df_second= ratings.loc[ratings.userId==person2]

    

    #Finding Similar Movies for person1 & person2 

    df= pd.merge(df_first,df_second,how='inner',on='movieId')

    

    #If no similar movie found, return 0 (No Similarity)

    if(len(df)==0): return 0

    

    #sum of squared difference between ratings

    sum_of_squares=sum(pow((df['rating_x']-df['rating_y']),2))

    return 1/(1+sum_of_squares)

    

# Checking working by passing similar ID, Corerelation should be 1

euclidean_distance(1,1) # Swwweeettt!!!
def pearson_score(person1,person2):

    

    #Get detail for Person1 and Person2

    df_first= ratings.loc[ratings.userId==person1]

    df_second= ratings.loc[ratings.userId==person2]

    

    # Getting mutually rated items    

    df= pd.merge(df_first,df_second,how='inner',on='movieId')

    

    # If no rating in common

    n=len(df)

    if n==0: return 0



    #Adding up all the ratings

    sum1=sum(df['rating_x'])

    sum2=sum(df['rating_y'])

    

    ##Summing up squares of ratings

    sum1_square= sum(pow(df['rating_x'],2))

    sum2_square= sum(pow(df['rating_y'],2))

    

    # sum of products

    product_sum= sum(df['rating_x']*df['rating_y'])

    

    ## Calculating Pearson Score

    numerator= product_sum - (sum1*sum2/n)

    denominator=sqrt((sum1_square- pow(sum1,2)/n) * (sum2_square - pow(sum2,2)/n))

    if denominator==0: return 0

    

    r=numerator/denominator

    

    return r



#Checking function by passing similar ID, Output should be 1

pearson_score(1,1)
# Returns the best matches for person from the prefs dictionary.

# Number of results and similarity function are optional params.

def topMatches(personId,n=5,similarity=pearson_score):

    scores=[(similarity(personId,other),other) for other in ratings.loc[ratings['userId']!=personId]['userId']]

    # Sort the list so the highest scores appear at the top

    scores.sort( )

    scores.reverse( )

    return scores[0:n]



topMatches(1,n=3) ## Getting 3 most similar Users for Example 
# Gets recommendations for a person by using a weighted average

# of every other user's rankings

def getRecommnedation(personId, similarity=pearson_score):

    '''

    totals: Dictionary containing sum of product of Movie Ratings by other user multiplied by weight(similarity)

    simSums: Dictionary containung sum of weights for all the users who have rated that particular movie.

    '''

    totals,simSums= {},{}

    

    df_person= ratings.loc[ratings.userId==personId]

    

    for otherId in ratings.loc[ratings['userId']!=personId]['userId']: # all the UserID except personID

        

        # Getting Similarity with OtherID

        sim=similarity(personId,otherId)

        

        # Ignores Score of Zero or Negatie correlation         

        if sim<=0: continue

            

        df_other=ratings.loc[ratings.userId==otherId]

        

        #Movies not seen by the personID

        movie=df_other[~df_other.isin(df_person).all(1)]

        

        for movieid,rating in (np.array(movie[['movieId','rating']])):

            #similarity* Score

            totals.setdefault(movieid,0)

            totals[movieid]+=rating*sim

            

            #Sum of Similarities

            simSums.setdefault(movieid,0)

            simSums[movieid]+=sim

            

        

        

        

        # Creating Normalized List

        ranking=[(t/simSums[item],item) for item,t in totals.items()]

        

        # return the sorted List

        ranking.sort()

        ranking.reverse()

        recommendedId=np.array([x[1] for x in ranking])

        

        

        return np.array(movies[movies['movieId'].isin(recommendedId)]['title'])[:20]
# Example Recoomendation

#returns 20 recommended movie for the given UserID

# userId can be ranged from 1 to 671

getRecommnedation(1)

getRecommnedation(671)
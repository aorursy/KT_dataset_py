import pandas as pd

def read_in_movie_preference():

    file_location = "./data/movie_preference.csv"

    df = None

    column_names = []

    preference = {}

    

    file_location = "../input/movie_preference.csv"  #this is mine path, not yours

#     df = pd.read_csv(file_location, index_col = 0)

    df = pd.read_csv(file_location)

    column_names = list(df.columns[1:])  #drop the first column

#     print(column_names)

#     print(df.values.tolist())

    for i in df.values.tolist():

        preference[i[0]] = i[1:]    #first column is the name

#     print(preference)

    return [df, column_names, preference]
[df, column_names, preference] = read_in_movie_preference()

assert df.shape == (186, 21)
[df, column_names, preference] = read_in_movie_preference()

assert column_names == ['The Shawshank Redemption', 'The Godfather',

                       'The Dark Knight ', 'Star Wars: The Force Awakens',

                       'The Lord of the Rings: The Return of the King',

                       'Inception', 'The Matrix ', 'Avengers: Infinity War ',

                       'Interstellar ', 'Spirited Away', 'Coco', 'The Dark Knight Rises',

                       'Braveheart', 'The Wolf of Wall Street', 'Gone Girl ', 'La La Land',

                       'Shutter Island', 'Ex Machina', 'The Martian', 'Kingsman: The Secret Service']
[df, column_names, preference] = read_in_movie_preference()

assert preference["DJZ"] == [0, 1, 1, 0, 1, 1, 1, -1, 1, 1, 0, -1, -1, -1, 1, -1, 1, -1, 1, -1]
def movies_popularity_ranking(df, movie_names):

    movie_popularity_rank = []

    

    movie_scores = []

    for movie_name in movie_names:

        likes,dislikes = (0,0)

        score = df[movie_name].sum()

        for i in df[movie_name]:

            if i==1:

                likes+=1

            elif i==-1:

                dislikes+=1

        assert(score==likes-dislikes)    #for verify

        movie_scores.append([movie_name,score])    #calculate the score, and store as a list of list [movie,score]

        print(movie_name,score,likes,dislikes)

    print(movie_scores)

    sorted_movie_scores = sorted(movie_scores,key=lambda k:k[1], reverse=True)    #use sorted function to sort the list based on k[1], which stands for the score

    print(sorted_movie_scores)

    

    movie_rank = {}    #use dict of movie:rank to store the rank info.

    for i in range(len(sorted_movie_scores)):

        rank = len(movie_rank)+1

        if(sorted_movie_scores[i][1]==sorted_movie_scores[i-1][1]):

            movie_rank[sorted_movie_scores[i][0]] = movie_rank[sorted_movie_scores[i-1][0]]    #if this movie's score equals the previous one, than it has the same rank. (called tie)

        else:

            movie_rank[sorted_movie_scores[i][0]] = rank    #for no tied movies, the rank equals to the current length of movie_rank plus one

    print(movie_rank)

    

    movie_popularity_rank = [movie_rank[i] for i in movie_names]  #generate the rank list from movie_rank

    return movie_popularity_rank
[df, movie_names, preference] = read_in_movie_preference()

movie_popularity_rank = movies_popularity_ranking(df, movie_names)

assert movie_popularity_rank == [2, 9, 7, 19, 9, 1, 13, 7, 5, 11, 3, 15, 18, 12, 14, 6, 17, 20, 16, 4]
def Recommendation(movie_popularity_ranking, preference, movie_names, name):

    recommended_movie = ""

    

    watch_history = preference[name]

    print(watch_history)

    movie_not_watched_rank = []

    for i in range(len(watch_history)):

        if watch_history[i]==0:

            movie_not_watched_rank.append([movie_names[i],movie_popularity_ranking[i]])

    print(movie_not_watched_rank)

    sorted_movie_not_watched_rank = sorted(movie_not_watched_rank, key=lambda k:k[1])

    recommended_movie = sorted_movie_not_watched_rank[0][0]

    return recommended_movie

[df, movie_names, preference] = read_in_movie_preference()

movie_popularity_rank = movies_popularity_ranking(df, movie_names)

assert Recommendation(movie_popularity_rank, preference, movie_names, "DJZ") == 'The Shawshank Redemption'
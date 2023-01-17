import numpy as np

import pandas as pd



#reading the artists with genres file:

df=pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv")

df.artists.str.strip()

df = df.drop_duplicates(subset='artists',keep=False)



#removing attributes that are not required for recommending:

df.drop(['count','duration_ms','key','mode','genres'],axis=1,inplace=True)



#Standardizing columns so that the min value is zero and max value is 1:

df['tempo'] = df['tempo']/217.743000

df['loudness'] = (df['loudness'] + 60)/61.342

df['liveness'] = df['liveness']/0.991

df['acousticness'] = df['acousticness']/0.996

df['danceability'] = df['danceability']/0.986

df['speechiness'] = df['speechiness']/0.964

df['valence'] = df['valence']/0.999

df['popularity'] = df['popularity']/97



#converting df from wide to long:

df2=pd.melt(df,id_vars=['artists'],var_name='song_features', value_name='values')



#creating a table with each artist as a separate column and each row as a separate attribute:

art_mat=df2.pivot_table(index="song_features",columns="artists",values="values")
#function for getting user input of preferred artists:

def getlist(): 

    arts = []

    print("Enter name of a preferred artist: ")

    inp = input()

    arts.append(inp)

    

    while inp != "":

        print("Enter name of any additional preferred artists or hit ENTER to continue: ")

        inp = input()

        arts.append(inp)

    

    arts.remove("")

    

    print(f"Your list of preferred artists is as follows: {arts}")



    return arts
#function for returning recommended artists:

def recommend(artist_list,artist_matrix): 

    #artist_list = getlist() #uncomment this if you want to get live user input

    #remove artist_list from the function definition if you are using getlist to get user input

    art_corr_df = pd.DataFrame()

    count = 0

    for artist in artist_list:

        

        if artist in art_mat.columns:

            #print(f"Artist: {artist}")

            artist_values = artist_matrix[artist]

            similar_to_art = artist_matrix.corrwith(artist_values)

            similar_to_art = pd.DataFrame(similar_to_art,columns=['Correlation'])

            similar_to_art = similar_to_art.sort_values(by="Correlation",ascending=False).head(10)

            art_corr_df = art_corr_df.append(similar_to_art)

        

        else:

            print(f"{artist} does not exist in the dataset and will not be included in the recommendations")

            count = count + 1

    

    if len(artist_list)==count:

        

        return print("None of the artists exist in the dataset. Try again.")

    

    else:

        art_corr_df.dropna(inplace=True)

        art_corr_df.reset_index(inplace=True)

        art_corr_df = art_corr_df.drop_duplicates(subset='artists',keep = False)

        art_corr_df = art_corr_df[art_corr_df.Correlation != 1.000000]

        art_corr_df = art_corr_df.sort_values(by="Correlation",ascending = False)

        art_corr_df.reset_index(inplace=True)

        art_corr_df = art_corr_df.drop('index',axis=1)

        

        return art_corr_df
artist_list = ['Ed Sheeran','Metallica']

recommend(artist_list,art_mat)
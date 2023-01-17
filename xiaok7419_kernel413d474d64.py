import pandas as pd

def read_in_movie_preference():

    file_location = "./data/movie_preference.csv"

    df = None

    column_names = []

    preference = {}

    

    file_location = "../input/movie-preferencecsv/movie_preference.csv"  #this is mine path, not yours

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
assert column_names == ['The Shawshank Redemption', 'The Godfather',

                       'The Dark Knight ', 'Star Wars: The Force Awakens',

                       'The Lord of the Rings: The Return of the King',

                       'Inception', 'The Matrix ', 'Avengers: Infinity War ',

                       'Interstellar ', 'Spirited Away', 'Coco', 'The Dark Knight Rises',

                       'Braveheart', 'The Wolf of Wall Street', 'Gone Girl ', 'La La Land',

                       'Shutter Island', 'Ex Machina', 'The Martian', 'Kingsman: The Secret Service']
assert preference["DJZ"] == [0, 1, 1, 0, 1, 1, 1, -1, 1, 1, 0, -1, -1, -1, 1, -1, 1, -1, 1, -1]
def jaccard_similarity(preference_1, preference_2):

    js = 0

    

    both_like = [preference_1[i]==1 and preference_2[i]==1 for i in range(len(preference_1))]

    at_least_1_like = [preference_1[i]==1 or preference_2[i]==1 for i in range(len(preference_1))]

#     print(both_like)

#     print(at_least_1_like)

    js = both_like.count(True) / at_least_1_like.count(True)

    return js
assert round(jaccard_similarity([1, 0, 1, -1], [1, 1, 0, 0]), 2) == 0.33

assert jaccard_similarity(preference["123"], preference["DJZ"]) == 0.25
def Find_Soul_Mate(preference, name):

    soulmate = ""

    soulmate_preference = []

    max_js = 0



    js = []

    for name_i in preference.keys():

        if(name == name_i):

            pass

        else:

            js.append([jaccard_similarity(preference[name],preference[name_i]),name_i])

            

    max_js = sorted(js, reverse=True)[0][0]

    soulmates = [x[1] for x in js if x[0]==max_js]

    soulmate = sorted(soulmates)[0]

    soulmate_preference = preference[soulmate]

    return [soulmate, soulmate_preference, max_js]

[soulmate, soulmate_preference, js] = Find_Soul_Mate(preference, "Rachel")

[soulmate, soulmate_preference, js]

assert soulmate == 'Kristen Xin'

assert js == 0.75
def Recommendation(preference, name, movie_names):

    recommendation = ""



    [soulmate, soulmate_preference, js] = Find_Soul_Mate(preference, name)

    for i in range(len(movie_names)):

        if(soulmate_preference[i]!=0 and preference[name][i]==0):

            return movie_names[i]

    

    return recommendation

assert Recommendation(preference, "DJZ", column_names) == 'The Shawshank Redemption'
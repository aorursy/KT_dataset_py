import pandas as pd

import ast

from scipy.spatial.distance import cosine, euclidean, hamming

from sklearn.preprocessing import normalize

from keras.preprocessing import image

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from time import time
recipe = pd.read_csv('/kaggle/input/foodrecsysv1/raw-data_recipe.csv')

recipe = recipe.drop(columns=['image_url', 'cooking_directions', 'reviews'])

recipe.head()
# round average rating into 2 decimal places

def avg_rate(col):

    return f'{col:.2f}'
recipe.aver_rate = recipe.aver_rate.apply(avg_rate)

recipe.head()
recipe.aver_rate = recipe.aver_rate.astype(float)

recipe.dtypes
recipe.shape
# total number of unique recipe

recipe.recipe_id.nunique()
recipe.nutritions[0]
# turn nutritions data from string to dictionary

list_of_dict = []



for row in recipe.nutritions:

    list_of_dict.append(ast.literal_eval(row))
# extract percent daily values for selected nutritions

calories_list = []

fat_list = []

carbohydrates_list = []

protein_list = []

cholesterol_list = []

sodium_list = []

fiber_list = []



for x in range(len(list_of_dict)):

    calories_list.append(list_of_dict[x]['calories']['percentDailyValue'])

    fat_list.append(list_of_dict[x]['fat']['percentDailyValue'])

    carbohydrates_list.append(list_of_dict[x]['carbohydrates']['percentDailyValue'])

    protein_list.append(list_of_dict[x]['protein']['percentDailyValue'])

    cholesterol_list.append(list_of_dict[x]['cholesterol']['percentDailyValue'])

    sodium_list.append(list_of_dict[x]['sodium']['percentDailyValue'])

    fiber_list.append(list_of_dict[x]['fiber']['percentDailyValue'])
# group all the data into dataframe

data = {'calories': calories_list, 'fat': fat_list, 'carbohydrates': carbohydrates_list, 

       'protein': protein_list, 'cholesterol': cholesterol_list, 'sodium': sodium_list, 

       'fiber': fiber_list}



df = pd.DataFrame(data)

df.index = recipe['recipe_id']

df.head()
df.isnull().sum()
print(df.shape)

df = df.dropna()

df.shape
# impute string data into numeric value

def text_cleaning(cols):

    if cols == '< 1':

        return 1

    else:

        return cols
for col in df.columns:

    df[col] = df[col].apply(text_cleaning)
df.head()
df.isna().sum()
df.dtypes
df = df.apply(pd.to_numeric)

df.dtypes
# normalized nutrition data by columns

df_normalized = pd.DataFrame(normalize(df, axis=0))

df_normalized.columns = df.columns

df_normalized.index = df.index

df_normalized.head()
df_normalized.tail()
# show recipe id, recipe name and image of selected recipe

def selected_recipe(recipe_id):

    image_path = "/kaggle/input/foodrecsysv1/raw-data-images/raw-data-images/{}.jpg"

    image_path = image_path.format(recipe_id)

    

    img = image.load_img(image_path)

    img = image.img_to_array(img, dtype='int')

        

    fig, ax = plt.subplots(1,1)

    ax.imshow(img)

    ax.axis('off')

    

    recipe_df = recipe.set_index('recipe_id')

    x = "{}  {}".format(recipe_id, recipe_df.at[recipe_id, 'recipe_name'])

    ax.set_title(x)
"""

Nutrition Recommender based on different distance calculation approaches



df_normalized: normalized nutrition data

distance_method: distance calculation approach: e.g. cosine, euclidean, hamming

recipe_id: find similar recipes based on the selected recipe

N: Top N recipe(s)



return 1) nutrition data of selected recipe and Top N recommendation, 

2) recipe id, recipe name and image of Top N recommendation

"""



def nutrition_recommender(distance_method, recipe_id, N):

    start = time()

    

    allRecipes = pd.DataFrame(df_normalized.index)

    allRecipes = allRecipes[allRecipes.recipe_id != recipe_id]

    allRecipes["distance"] = allRecipes["recipe_id"].apply(lambda x: distance_method(df_normalized.loc[recipe_id], df_normalized.loc[x]))

    TopNRecommendation = allRecipes.sort_values(["distance"]).head(N).sort_values(by=['distance', 'recipe_id'])

    # sort by distance then recipe id, the smaller value of recipe id will be picked. 

    

    recipe_df = recipe.set_index('recipe_id')

    recipe_id = [recipe_id]

    recipe_list = []

    image_list = []

    image_path = "/kaggle/input/foodrecsysv1/raw-data-images/raw-data-images/{}.jpg"

    for recipeid in TopNRecommendation.recipe_id:

        recipe_id.append(recipeid)   # list of recipe id of selected recipe and recommended recipe(s)

        recipe_list.append("{}  {}".format(recipeid, recipe_df.at[recipeid, 'recipe_name']))

        image_list.append(image_path.format(recipeid))

    

    image_array = []

    for imagepath in image_list:

        img = image.load_img(imagepath)

        img = image.img_to_array(img, dtype='int')

        image_array.append(img)

        

    fig = plt.figure(figsize=(15,15))

    gs1 = gridspec.GridSpec(1, N)

    axs = []

    for x in range(N):

        axs.append(fig.add_subplot(gs1[x]))

        axs[-1].imshow(image_array[x])

    [axi.set_axis_off() for axi in axs]

    for axi, x in zip(axs, recipe_list):

        axi.set_title(x)

    

    end = time()

    running_time = end - start

    print('time cost: %.5f sec' %running_time)

    return df_normalized.loc[recipe_id, :]
selected_recipe(222388)
nutrition_recommender(cosine, 222388, 3)
nutrition_recommender(euclidean, 222388, 3)
nutrition_recommender(hamming, 222388, 3)
"""

Hybrid Nutrition Recommender which integrates Top 2 recommendation from 3 different distance approaches 

(cosine, euclidean, hamming) and sort the results by selected criteria(s)



df_normalized: normalized nutrition data

recipe_id: find similar recipes based on the selected recipe

sort_order: must be in list, 4 options available: ['aver_rate'], ['review_nums'], ['aver_rate', 'review_nums'], ['review_nums', 'aver_rate']

N: Top N recipe(s)



return 1) recipe id, recipe name and image of Top N recommendation, 

2) nutrition data of selected recipe and Top N recommendation,

3) average rating and number of review of Top N recommendation

"""



def nutrition_hybrid_recommender(recipe_id, sort_order, N):

    start = time()

    

    allRecipes_cosine = pd.DataFrame(df_normalized.index)

    allRecipes_cosine = allRecipes_cosine[allRecipes_cosine.recipe_id != recipe_id]

    allRecipes_cosine["distance"] = allRecipes_cosine["recipe_id"].apply(lambda x: cosine(df_normalized.loc[recipe_id], df_normalized.loc[x]))

    

    allRecipes_euclidean = pd.DataFrame(df_normalized.index)

    allRecipes_euclidean = allRecipes_euclidean[allRecipes_euclidean.recipe_id != recipe_id]

    allRecipes_euclidean["distance"] = allRecipes_euclidean["recipe_id"].apply(lambda x: euclidean(df_normalized.loc[recipe_id], df_normalized.loc[x]))

    

    allRecipes_hamming = pd.DataFrame(df_normalized.index)

    allRecipes_hamming = allRecipes_hamming[allRecipes_hamming.recipe_id != recipe_id]

    allRecipes_hamming["distance"] = allRecipes_hamming["recipe_id"].apply(lambda x: hamming(df_normalized.loc[recipe_id], df_normalized.loc[x]))

    

    Top2Recommendation_cosine = allRecipes_cosine.sort_values(["distance"]).head(2).sort_values(by=['distance', 'recipe_id'])

    Top2Recommendation_euclidean = allRecipes_euclidean.sort_values(["distance"]).head(2).sort_values(by=['distance', 'recipe_id'])

    Top2Recommendation_hamming = allRecipes_hamming.sort_values(["distance"]).head(2).sort_values(by=['distance', 'recipe_id'])

    

    recipe_df = recipe.set_index('recipe_id')

    hybrid_Top6Recommendation = pd.concat([Top2Recommendation_cosine, Top2Recommendation_euclidean, Top2Recommendation_hamming])

    aver_rate_list = []

    review_nums_list = []

    for recipeid in hybrid_Top6Recommendation.recipe_id:

        aver_rate_list.append(recipe_df.at[recipeid, 'aver_rate'])

        review_nums_list.append(recipe_df.at[recipeid, 'review_nums'])

    hybrid_Top6Recommendation['aver_rate'] = aver_rate_list

    hybrid_Top6Recommendation['review_nums'] = review_nums_list

    TopNRecommendation = hybrid_Top6Recommendation.sort_values(by=sort_order, ascending=False).head(N).drop(columns=['distance'])

    

    recipe_id = [recipe_id]   

    recipe_list = []

    for recipeid in TopNRecommendation.recipe_id:

        recipe_id.append(recipeid)   # list of recipe id of selected recipe and recommended recipe(s)

        recipe_list.append("{}  {}".format(recipeid, recipe_df.at[recipeid, 'recipe_name']))

    

    image_list = []

    image_path = "/kaggle/input/foodrecsysv1/raw-data-images/raw-data-images/{}.jpg"

    for recipeid in TopNRecommendation.recipe_id:

        image_list.append(image_path.format(recipeid))

    

    image_array = []

    for imagepath in image_list:

        img = image.load_img(imagepath)

        img = image.img_to_array(img, dtype='int')

        image_array.append(img)

        

    fig = plt.figure(figsize=(15,15))

    gs1 = gridspec.GridSpec(1, N)

    axs = []

    for x in range(N):

        axs.append(fig.add_subplot(gs1[x]))

        axs[-1].imshow(image_array[x])

    [axi.set_axis_off() for axi in axs]

    for axi, x in zip(axs, recipe_list):

        axi.set_title(x)

    

    end = time()

    running_time = end - start

    print('time cost: %.5f sec' %running_time)

    return df_normalized.loc[recipe_id, :], TopNRecommendation
selected_recipe(222886)
nutrition_ar, topN_ar = nutrition_hybrid_recommender(222886, ['aver_rate'], 3)
nutrition_ar
topN_ar
nutrition_rn, topN_rn = nutrition_hybrid_recommender(222886, ['review_nums'], 3)
nutrition_rn
topN_rn
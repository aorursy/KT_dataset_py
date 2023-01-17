import numpy as np

import pandas as pd

import pandas_profiling as pp
df_coffee_reviews_arabica = pd.read_csv("../input/coffee-quality-database-from-cqi/arabica_data_cleaned.csv")



df_coffee_reviews_arabica.head(10)
df_coffee_reviews_robusta = pd.read_csv("../input/coffee-quality-database-from-cqi/robusta_data_cleaned.csv")



df_coffee_reviews_robusta.head(10)
df_coffee_reviews_arabica.drop(columns=[df_coffee_reviews_arabica.columns[0]], inplace=True)

df_coffee_reviews_robusta.drop(columns=[df_coffee_reviews_robusta.columns[0]], inplace=True)
df_coffee_reviews_arabica.columns
df_coffee_reviews_robusta.columns
df_coffee_reviews_arabica.columns == df_coffee_reviews_robusta.columns
df_coffee_reviews_arabica.columns.difference(df_coffee_reviews_robusta.columns)
df_coffee_reviews_robusta.rename(

    columns={

        "Salt...Acid": "Acidity",

        "Fragrance...Aroma": "Aroma",

        "Bitter...Sweet": "Sweetness",

        "Uniform.Cup": "Uniformity",

        "Mouthfeel": "Body"

    },

    inplace=True

)
df_coffee_reviews_arabica.columns.difference(df_coffee_reviews_robusta.columns)
pp.ProfileReport(df_coffee_reviews_arabica)
pp.ProfileReport(df_coffee_reviews_robusta)
df_coffee_reviews = pd.concat(

    [df_coffee_reviews_arabica, df_coffee_reviews_robusta],

    ignore_index=True,

    sort=False

)



df_coffee_reviews.head(10)
pp.ProfileReport(df_coffee_reviews)
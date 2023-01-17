import gc

import os

from time import time

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sys

from tqdm import tqdm_notebook, tqdm

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

##############################################

%matplotlib inline
DATA_PATH = "/Users/marievachelard/Downloads/innovationcup/"

for dirname, _, filenames in os.walk(DATA_PATH):

    for filename in filenames:

        print(os.path.join(dirname, filename))
ID_Data_train = pd.read_csv(os.path.join(DATA_PATH,"ID_Data_train.csv"))

ID_Data_test = pd.read_csv(os.path.join(DATA_PATH,"ID_Data_test.csv"))

ID_Time_train = pd.read_csv(os.path.join(DATA_PATH,"ID_Time_train.csv"))
print(ID_Data_train.shape)

print(ID_Data_train.columns)

print(ID_Data_train['id'].nunique())

ID_Data_train.sample(3)
def get_mapping_id_race_id(train, test):

    """

    train : DF comprenant les courses des bateaux (séries temporelles des variables considérées)

    test : DF comprenant les courses des bateaux (séries temporelles des variables considérées)

    result : DF comprenant les bateaux du train et du test avec leur race_id

    """

    df = pd.concat([train, test], axis=0)

    return df[['id_race', 'id']].drop_duplicates().reset_index(drop=True)



map_id_race_id = get_mapping_id_race_id(ID_Data_train, ID_Data_test)
def calc_diff_angle(data):

    """

    data : DF comprenant les courses des bateaux (séries temporelles des variables considérées)

    dont les features direction_vent et cap

    result : DF comprenant les courses des bateaux (séries temporelles des variables considérées)

    dont le feature Diff_angle entre la direction du vent et le cap

    """

    data.loc[:,'Diff_angle'] = data.loc[:,'direction_vent'] - data.loc[:,'cap']

    return data
def creer_features(input_data):

    """

    input_data : DF comprenant les courses des bateaux (séries temporelles des variables considérées)

    time_id : DF lié à input_data qui comprend l'ID, le temps, la course, et le rang 

    """

    X_model = pd.DataFrame()

    IDs = input_data['id'].drop_duplicates().values

    data = input_data.copy()

    for i in tqdm_notebook(IDs): 

        data_id = data[data['id']==i]

        data_id = calc_diff_angle(data_id)

        X_model.loc[i, 'lat_mean'] = data_id['latitude'].mean()

        X_model.loc[i, 'long_std'] = data_id['longitude'].std()

        

    X_model = X_model.fillna(0)

    return X_model



X_train = creer_features(ID_Data_train)

y_train = X_train.merge(ID_Time_train, left_index=True, right_on='id', how='left')['temps']

print(len(y_train), X_train.shape)

X_test = creer_features(ID_Data_test)
def rmse(y, y_pred):

    return np.sqrt(np.mean(np.square(y - y_pred)))
def apk(actual, predicted, k=10):

    """

    Computes the average precision at k.

    This function computes the average prescision at k between two lists of

    items.

    Parameters

    ----------

    actual : list

             A list of elements that are to be predicted (order doesn't matter)

    predicted : list

                A list of predicted elements (order does matter)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The average precision at k over the input lists

    """

    if len(predicted)>k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i,p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i+1.0)



    if not actual:

        return 0.0



    return score / min(len(actual), k)





def mapk(actual, predicted, k=10):

    """

    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists

    of lists of items.

    Parameters

    ----------

    actual : list

             A list of lists of elements that are to be predicted

             (order doesn't matter in the lists)

    predicted : list

                A list of lists of predicted elements

                (order matters in the lists)

    k : int, optional

        The maximum number of predicted elements

    Returns

    -------

    score : double

            The mean average precision at k over the input lists

    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
def convert_to_kaggle_submission_format(X, y, mapping):

    """

    Transform a dataset of boats and its feature, its target or prediction and a race_id-boat_id

    mapping into the format for kaggle submission

    Parameters

    ----------

    X : pandas DataFrame

        DataFrame of boats features and id_boat

    y : list

        A list of (predicted or true) race duration for a boat

    mapping : pandas DataFrame

        DataFrame with the mapping id_race-id_boat

    Returns

    -------

    result : pandas DataFrame

            A pandas DataFrame with a column id_race, and a column ranking, with the ranking of

            the race

    """



    data = (

        X

        .copy()

        .assign(temps=list(y))

        .merge(

            mapping,

            left_index=True,

            right_on='id',

            how='left')

        [['id_race', 'id', 'temps']]

        )



    data = (

        data.sort_values(by=["id_race", "temps"])

        .reset_index(drop=True)

        .groupby("id_race")

        .id.apply(list)

        .reset_index()

        )



    return data.assign(ranking=data.id.apply(lambda x: " ".join(x))).loc[

            :, ["id_race", "ranking"]

        ]
LR = LinearRegression()

LR.fit(X_train, y_train)



y_pred_train = LR.predict(X_train)



rmse(y_train, y_pred_train)



raw_train_for_kaggle_sub = convert_to_kaggle_submission_format(X_train, y_train, map_id_race_id)

pred_train_for_kaggle_sub = convert_to_kaggle_submission_format(X_train, y_pred_train, map_id_race_id)



mapk(

    list(raw_train_for_kaggle_sub.ranking.apply(lambda x: x.split(' ')[:3])), list(pred_train_for_kaggle_sub.ranking.apply(lambda x: x.split(' '))),

    k=10

    )



y_pred_test = LR.predict(X_test)

pred_test_for_kaggle_sub = convert_to_kaggle_submission_format(X_test, y_pred_test, map_id_race_id)



pred_test_for_kaggle_sub.to_csv("soumission_finale_TEAMNAME#3.csv", index=False)
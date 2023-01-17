import pandas as pd

import xgboost as xgb

import numpy as np



#useful for displaying wide data frames

pd.set_option('display.max_columns', 50)
#load the data into Pandas dataframes

df_market = pd.read_csv("../input/markets.csv")

df_runners = pd.read_csv("../input/runners.csv",dtype={'barrier': np.int16,'handicap_weight': np.float16})



#for my simple model, I'm ignoring other columns. I recommend starting with form if you're looking to add features

#df_odds = pd.read_csv("../input/odds.csv")

#df_form = pd.read_csv("../input/forms.csv")

#df_condition = pd.read_csv("../input/conditions.csv")

#df_weather = ("../input/weather.csv")

#df_rider = ("../input/riders.csv")

#df_horse = ("../input/horses.csv")

#df_horse_sex = ("../input/horse_sexes.csv")
#look at the first fives rows of the market table

df_market[0:5]
#look at the first fives rows of the runners table

df_runners[0:5]
#explore the barriers feature: does it look like it impacts chances of victory?

winners_by_barrier = df_runners[df_runners['position'] == 1][['id','barrier']].groupby('barrier').agg(['count'])

barrier_count = df_runners[['id','barrier']].groupby('barrier').agg(['count'])

pct_winner_by_barrier = winners_by_barrier/barrier_count[barrier_count.index.isin(winners_by_barrier.index)]

ax = pct_winner_by_barrier.plot(kind='bar')

ax.set_ylabel("Win Percentage")



#this notebook pushes up against memory limits. So I'm aggressive with garbage collection.

del winners_by_barrier, barrier_count, pct_winner_by_barrier
#explore weight: does it looks like it has an impact?

winners_by_weight = df_runners[df_runners['position'] == 1][['id','handicap_weight']].groupby('handicap_weight').agg(['count'])

winners_by_weight = winners_by_weight[winners_by_weight > 30].dropna()

weight_count = df_runners[['id','handicap_weight']].groupby('handicap_weight').agg(['count'])

pct_winners_by_weight = winners_by_weight/weight_count[weight_count.index.isin(winners_by_weight.index)]

ax = pct_winners_by_weight.plot(kind='bar')

ax.set_ylabel("Win Percentage")

del winners_by_weight, weight_count, pct_winners_by_weight
#explore weight: does it looks like it has an impact?

winners_by_rider = df_runners[df_runners['position'] == 1][['id','rider_id']].groupby('rider_id').agg(['count'])

#only inclide riders who have more than 10 races

winners_by_rider =winners_by_rider[winners_by_rider > 10].dropna()

rider_count = df_runners[['id','rider_id']].groupby('rider_id').agg(['count'])

rider_count = rider_count[rider_count.index.isin(winners_by_rider.index)]

pct_winners_by_rider = winners_by_rider/rider_count

pct_winners_by_rider.columns = ['Win_Percentage']

pct_winners_by_rider = pct_winners_by_rider.sort_values(by='Win_Percentage',ascending=False)

ax = pct_winners_by_rider.plot(kind='bar')

ax.set_ylabel("Win Percentage")

del winners_by_rider, rider_count, pct_winners_by_rider
##merge the runners and markets data frames

df_runners_and_market = pd.merge(df_runners,df_market,left_on='market_id',right_on='id',how='outer')

df_runners_and_market.index = df_runners_and_market['id_x'] 
numeric_features = ['position','market_id','barrier','handicap_weight']

categorical_features = ['rider_id']



#convert to factors

for feature in categorical_features:

    df_runners_and_market[feature] = df_runners_and_market[feature].astype(str)

    df_runners_and_market[feature] = df_runners_and_market[feature].replace('nan','0') #have to do this because of a weird random forest bug



    df_features = df_runners_and_market[numeric_features]



for feature in categorical_features:

    encoded_features = pd.get_dummies(df_runners_and_market[feature])

    encoded_features.columns = feature + encoded_features.columns

    df_features = pd.merge(df_features,encoded_features,left_index=True,right_index=True,how='inner') 



#turn the target variable into a binary feature: did or did not win

df_features['win'] = False

df_features.loc[df_features['position'] == 1,'win'] = True



#del df_runners_and_market, encoded_features, df_features['position']
training_races = np.random.choice(df_features['market_id'].unique(),size=int(round(0.7*len(df_features['market_id'].unique()),0)),replace=False)

df_train = df_features[df_features['market_id'].isin(training_races)]

df_test = df_features[~df_features['market_id'].isin(training_races)]



#del df_features
gbm = xgb.XGBClassifier(objective='binary:logistic').fit(df_train.drop(df_train[['win','position','market_id']],axis=1)

, df_train['win'])

predictions = gbm.predict_proba(df_test.drop(df_test[['win','position','market_id']],axis=1))[:,0]

df_test['predictions'] = predictions

df_test = df_test[['predictions','win','market_id']]

#del df_train



print( "total predictions:", len( predictions ), "\nunique values:", len( np.unique( predictions )))
from matplotlib import pyplot as plt

res = plt.hist( predictions, bins = 20 )
counter = 0

more_than_one_counter = 0



def inspect_predictions(df):

    global counter, more_than_one_counter

    max_p = df.predictions.max()

    max_p_count = (df.predictions == max_p).sum()

    if max_p_count > 1:

        more_than_one_counter += 1

    counter += 1    

    

df_test.groupby("market_id").apply(inspect_predictions)

print( "total races:", counter )

print ( "races with duplicate max value:", more_than_one_counter, 

    "({:.0%})".format( more_than_one_counter / float( counter )))
df_odds = pd.read_csv("../input/odds.csv")

df_odds = df_odds[df_odds['runner_id'].isin(df_test.index)]



#I take the mean odds for the horse rather than the odds 1 hour before or 10 mins before. You may want to revisit this.

average_win_odds = df_odds.groupby(['runner_id'])['odds_one_win'].mean()



#delete when odds are 0 because there is no market for this horse

average_win_odds[average_win_odds == 0] = np.nan

df_test['odds'] = average_win_odds

df_test = df_test.dropna(subset=['odds'])

#given that I predict multiple winners, there's leakage if I don't shuffle the test set (winning horse appears first and I put money on the first horse I predict to win)

#df_test = df_test.iloc[np.random.permutation(len(df_test))]


#select the horse I picked as most likely to win

df_profit = df_test.loc[df_test.groupby("market_id")["predictions"].idxmax()]

df_profit

investment = 0

payout = 0

for index, row in df_profit.iterrows():

    investment +=1

    

    if (row['win']):

        payout += row['odds']



investment_return = round((payout - investment)/investment*100,2)

print("This algorithm and betting system will generate a " + str(investment_return) + "% return\n")

print("Note: you can't read much from a single run. Best to setup a cross validation framework and look at the return over many runs")
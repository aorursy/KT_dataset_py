import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/chess/games.csv')
df.head()
df.winner.value_counts()
print('White appears to win', round(10001 / 20058, 2) * 100, '% of the games')

print('while Black wins', round(9107 / 20058, 2) * 100, '% of the games')

print('and', round(950 / 20058, 2) * 100, '% of the games are a draw')
#This cell creates a dataframe where openings are aggregated by the number of wins for each side

open_df = df.groupby(by='opening_name').winner.value_counts()

open_df = open_df.reset_index(name='wins')

open_df = open_df.sort_values(by='wins', ascending=False)
#The dataframes are splits by whether black or white won

black_wins = open_df[open_df['winner'] == 'black']

white_wins = open_df[open_df['winner'] == 'white']
black_wins.head()
white_wins.head()
#This cell takes the top 5 openings for each side and gets the amount of wins as a percentage

black_winner = list(black_wins.head().opening_name)

white_winner = list(white_wins.head().opening_name)

winner = black_winner + white_winner

dataframes = []

for x in winner:

    temp = open_df[open_df['opening_name'] == x]

    temp['sum'] = temp.wins.sum().astype(int)

    temp['percentage'] = temp['wins'] / temp['sum']

    dataframes.append(temp)

win_prob = dataframes[0]

for x in dataframes[1:]:

    win_prob = pd.concat([win_prob, x])
x = win_prob[win_prob['winner'] == 'black'].opening_name

y = win_prob[win_prob['winner'] == 'black'].percentage

plt.figure(dpi=100)

plt.bar(x, height=y, edgecolor='black')

plt.xticks(rotation='vertical')

plt.title('Win percentage by opening for black')

plt.xlabel('Opening')

plt.ylabel('Percentage(Out of 1)')
x = win_prob[win_prob['winner'] == 'white'].opening_name

y = win_prob[win_prob['winner'] == 'white'].percentage

plt.figure(dpi=100)

plt.bar(x, height=y, edgecolor='black')

plt.xticks(rotation='vertical')

plt.title('Win percentage by opening for white')

plt.xlabel('Opening')

plt.ylabel('Percentage(Out of 1)')
#This cell puts white id and black id under a common column and puts them into one dataframe

mask = ['white_id', 'white_rating']

white_player_df = df[mask]

white_player_df.columns = ['player_id', 'player_rating']

mask = ['black_id', 'black_rating']

black_player_df = df[mask]

black_player_df = black_player_df[mask]

black_player_df.columns = ['player_id', 'player_rating']

player_df = pd.concat([white_player_df, black_player_df])
#This cell takes the max rating for each id

ID = []

rating = []

for x in player_df.player_id.unique():

    ID.append(x)

    temp = player_df[player_df['player_id'] == x]

    rating.append(temp.player_rating.max())

player_df = pd.DataFrame()

player_df['player_id'] = np.array(ID)

player_df['player_rating'] = np.array(rating)
top_10 = player_df.sort_values(by='player_rating', ascending=False).head(10)

plt.figure(dpi=100)

plt.barh(y=top_10.player_id, width=top_10.player_rating, edgecolor='black')



plt.title('Top 10 players by rating')

plt.xlabel('Player Rating')

plt.ylabel('Player ID')
top_20 = player_df.sort_values(by='player_rating', ascending=False).head(20)

plt.figure(dpi=450)

plt.barh(y=top_20.player_id, width=top_20.player_rating, edgecolor='black')



plt.title('Top 20 players by rating')

plt.xlabel('Player Rating')

plt.ylabel('Player ID')
#This cell turns value counts into a dataframe

resolve = pd.DataFrame(df.victory_status.value_counts())

resolve = resolve.reset_index()
plt.figure(dpi=100)

plt.bar(x=resolve['index'], height=resolve.victory_status, edgecolor='black')

plt.title('Victory Status')

plt.xlabel('Status')

plt.ylabel('Number of occurances')
turn_df = df.groupby(by='victory_status').turns.mean().reset_index()

plt.figure(dpi=100)

plt.bar(x=turn_df.victory_status, height=turn_df.turns, edgecolor='black')

plt.title('Victory Status by average number of turns')

plt.xlabel('Status')

plt.ylabel('Number of turns')
#For each game, it takes the difference between the winner and the loser

difference = []

for x in range(df.shape[0]):

    temp = df.iloc[x]

    if temp['winner'] == 'white':

        diff = temp.white_rating - temp.black_rating

        difference.append(diff)

    elif temp['winner'] == 'black':

        diff = temp.black_rating - temp.white_rating

        difference.append(diff)

    else:

        ratings = [temp.black_rating, temp.white_rating]

        diff = max(ratings) - min(ratings)

        difference.append(diff)

df['Difference'] = np.array(difference)
mean_df = df.groupby(by='winner').Difference.mean().reset_index()

plt.figure(dpi=100)

plt.bar(x=mean_df['winner'], height=mean_df.Difference, edgecolor='black')

plt.title('Mean difference in ratings')

plt.xlabel('Winner')

plt.ylabel('Difference in rating')
mean_df = df.groupby(by='winner').Difference.std().reset_index()

plt.figure(dpi=100)

plt.bar(x=mean_df['winner'], height=mean_df.Difference, edgecolor='black')

plt.title('Standard deviation of difference in ratings')

plt.xlabel('Winner')

plt.ylabel('Difference in rating')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



cols = ['white_rating', 'black_rating']

X = df[cols]

y = df['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
lr = LogisticRegression()

lr.fit(X_train, y_train)
lr.score(X_test, y_test)
#Let's test this with a hypothetical match up with white rated 1463 and black rated 1500

test = np.array([1463, 1500])

test = test.reshape(1, -1)

print(lr.predict(test))

print(max(lr.predict_proba(test)[0]))

print((lr.predict_proba(test)[0]))
#Let's test this again with a far more distant rating

test = np.array([1686, 1523])

test = test.reshape(1, -1)

print(lr.predict(test))

print(max(lr.predict_proba(test)[0]))

print((lr.predict_proba(test)[0]))
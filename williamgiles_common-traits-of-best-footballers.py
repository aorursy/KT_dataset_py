# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import sqlite3
# creating the connection



conn = sqlite3.connect('../input/soccer/database.sqlite')
# have a look at the table types.I can see various items that i could join.



pd.read_sql("""

                SELECT * 

                FROM sqlite_master

                WHERE type='table'

                ;""",

                conn)
# attempting a join



top_players = pd.read_sql("""

                SELECT *

                FROM Player

                JOIN Player_Attributes

                ON Player.player_api_id = Player_attributes.player_api_id

                WHERE overall_rating > 80

                AND date>2015

                ;""",

                conn)
top_players
# it returns multiple rows for the same players, measured at different times. I want to drop the duplicates and have each players just once.



top_players = top_players.drop_duplicates(subset='player_api_id', keep='first')
filt = top_players['player_name'].str.contains('Neymar', na=False)

player = top_players[filt]
player.date
top_players



# i can see the number of rows has decreased, so there should only be row per player.
# I want to drop all the goalkeepers because they have different stats and may mess things up. 

# There is no goalkeeper position but I think I can do this by just taking any player with high goalkeeper ability. 

# This is enough to confirm to me that all these players are goalkeepers. So i will use this filter to drop them.



filt = top_players['gk_reflexes']>60

gks = top_players[filt]

gks
top_players.drop(top_players[filt].index, inplace=True)
# I now need to drop the GK related columns as I dont care about them. I will first get the column names



top_players.columns
top_players.drop(['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes', 'id', 'player_fifa_api_id', 'player_api_id', 'date'],axis=1, inplace=True)
top_players.columns



# ok, I only have outfield players left. Lets import matplotlib and do some plotting.
import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
top_players.dtypes
x = top_players['overall_rating']



a = top_players['reactions']

b = top_players['vision']

c = top_players['aggression']

d = top_players['positioning']

e = top_players['attacking_work_rate']

f = top_players['defensive_work_rate']
# I want to set the graphs to a certain size so I can see them clearly. 



from pylab import rcParams

rcParams['figure.figsize'] = 20, 10
fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3)



ax1.scatter(x,a)

ax1.set_title('Reactions')

ax1.set_ylim(ymin=50)

ax2.scatter(x,b)

ax2.set_title('Vision')

ax2.set_ylim(ymin=0)

ax3.scatter(x,c)

ax3.set_title('Aggression')

ax3.set_ylim(ymin=0)

ax4.scatter(x,d)

ax4.set_title('Positioning')

ax4.set_ylim(ymin=0)

ax5.hist(e)

ax5.set_title('attacking_work_rate')

ax6.hist(f)

ax6.set_title('defensive_work_rate')
# We can see the two standout players on the far right. Presumably messi and ronaldo, but I will check this.

# It would appear initially that reactions are the most important trait of the top players, along with positioning. 

# Aggression and defensive_work_rate would appear to not be important. 

filt = top_players['overall_rating']>92

top_two = top_players[filt]

top_two[['player_name','overall_rating']]
# So I can see that the top two are indeed messi and ronaldo. I will try dropping them and do it again



top_players_adj = top_players.drop(top_players[filt].index)
x = top_players_adj['overall_rating']



a = top_players_adj['reactions']

b = top_players_adj['vision']

c = top_players_adj['aggression']

d = top_players_adj['positioning']

e = top_players['attacking_work_rate']

f = top_players['defensive_work_rate']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)



ax1.scatter(x,a)

ax1.set_title('Reactions')

ax2.scatter(x,b)

ax2.set_title('Vision')

ax3.scatter(x,c)

ax3.set_title('Aggression')

ax4.scatter(x,d)

ax4.set_title('Positioning')

ax5.hist(e)

ax5.set_title('attacking_work_rate')

ax6.hist(f)

ax6.set_title('defensive_work_rate')
# ok, interestingly positioning doesn't seem to be that important, with some high level players having a lower score.

# There is some correlation but a lot of players are playing at a very high level without it



# there is a very clear trend in reactions and vision however.
top_players.columns
x = top_players['overall_rating']



a2 = top_players['height']

b2 = top_players['weight']

c2 = top_players['preferred_foot']

d2 = top_players['acceleration']

e2 = top_players['sprint_speed']

f2 = top_players['agility']

g2 = top_players['balance']

h2 = top_players['jumping']

i2 = top_players['stamina']

j2 = top_players['strength']

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5)



ax1.scatter(x,a2)

ax1.set_title('height')

ax2.scatter(x,b2)

ax2.set_title('weight')

ax3.hist(c2)

ax3.set_title('preferred_foot')

ax4.scatter(x,d2)

ax4.set_title('acceleration')

ax4.set_ylim(ymin=20)

ax5.scatter(x,e2)

ax5.set_title('sprint_speed')

ax5.set_ylim(ymin=20)

ax6.scatter(x,f2)

ax6.set_title('agility')

ax6.set_ylim(ymin=20)

ax7.scatter(x,g2)

ax7.set_title('balance')

ax7.set_ylim(ymin=20)

ax8.scatter(x,h2)

ax8.set_title('jumping')

ax8.set_ylim(ymin=20)

ax9.scatter(x,i2)

ax9.set_title('stamina')

ax9.set_ylim(ymin=20)

ax10.scatter(x,j2)

ax10.set_title('strength')

ax10.set_ylim(ymin=20)

# so there are certain things where there appears to be correlation. Acceleration, agility, sprint speed in particular. 

# perhaps surprisngly, stamina and balance are not too important although there is some correlation. 



#height, weight, jumping and stength seem largely irrelevant.



# preferred foot is obviously right. but 20% of top players prefer the left, which is above the average of 10%, suggesting left footedness is an advantage. 



# I will try again without the top two players.

x = top_players_adj['overall_rating']



a2 = top_players_adj['height']

b2 = top_players_adj['weight']

c2 = top_players_adj['preferred_foot']

d2 = top_players_adj['acceleration']

e2 = top_players_adj['sprint_speed']

f2 = top_players_adj['agility']

g2 = top_players_adj['balance']

h2 = top_players_adj['jumping']

i2 = top_players_adj['stamina']

j2 = top_players_adj['strength']
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2,5)



ax1.scatter(x,a2)

ax1.set_title('height')

ax2.scatter(x,b2)

ax2.set_title('weight')

ax3.hist(c2)

ax3.set_title('preferred_foot')

ax4.scatter(x,d2)

ax4.set_title('acceleration')

ax4.set_ylim(ymin=20)

ax5.scatter(x,e2)

ax5.set_title('sprint_speed')

ax5.set_ylim(ymin=20)

ax6.scatter(x,f2)

ax6.set_title('agility')

ax6.set_ylim(ymin=20)

ax7.scatter(x,g2)

ax7.set_title('balance')

ax7.set_ylim(ymin=20)

ax8.scatter(x,h2)

ax8.set_title('jumping')

ax8.set_ylim(ymin=20)

ax9.scatter(x,i2)

ax9.set_title('stamina')

ax9.set_ylim(ymin=20)

ax10.scatter(x,j2)

ax10.set_title('strength')

ax10.set_ylim(ymin=20)
# so it looks like acceleration and agility are the ones with the strongest correlation. 
top_players.columns
x = top_players['overall_rating']



a3 = top_players['crossing']

b3 = top_players['finishing']

c3 = top_players['heading_accuracy']

d3 = top_players['short_passing']

e3 = top_players['volleys']

f3 = top_players['dribbling']

g3 = top_players['curve']

h3 = top_players['long_passing']

i3 = top_players['ball_control']

j3 = top_players['shot_power']

k3 = top_players['long_shots']

l3 = top_players['interceptions']

m3 = top_players['marking']

n3 = top_players['standing_tackle']

o3 = top_players['sliding_tackle']

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5)



ax1.scatter(x,a3)

ax1.set_title('crossing')

ax2.scatter(x,b3)

ax2.set_title('finishing')

ax3.scatter(x,c3)

ax3.set_title('heading_accuracy')

ax4.scatter(x,d3)

ax4.set_title('short_passing')

ax4.set_ylim(ymin=50)

ax5.scatter(x,e3)

ax5.set_title('volleys')

ax5.set_ylim(ymin=0)

ax6.scatter(x,f3)

ax6.set_title('dribbling')

ax6.set_ylim(ymin=0)

ax7.scatter(x,g3)

ax7.set_title('curve')

ax7.set_ylim(ymin=0)

ax8.scatter(x,h3)

ax8.set_title('long_passing')

ax8.set_ylim(ymin=0)

ax9.scatter(x,i3)

ax9.set_title('ball_control')

ax9.set_ylim(ymin=40)

ax10.scatter(x,j3)

ax10.set_title('shot_power')

ax10.set_ylim(ymin=0)

ax11.scatter(x,k3)

ax11.set_title('long_shots')

ax11.set_ylim(ymin=0)

ax12.scatter(x,l3)

ax12.set_title('interceptions')

ax12.set_ylim(ymin=0)

ax13.scatter(x,m3)

ax13.set_title('marking')

ax13.set_ylim(ymin=0)

ax14.scatter(x,n3)

ax14.set_title('standing_tackle')

ax14.set_ylim(ymin=0)

ax15.scatter(x,o3)

ax15.set_title('sliding_tackle')

ax15.set_ylim(ymin=0)



x = top_players_adj['overall_rating']



a3 = top_players_adj['crossing']

b3 = top_players_adj['finishing']

c3 = top_players_adj['heading_accuracy']

d3 = top_players_adj['short_passing']

e3 = top_players_adj['volleys']

f3 = top_players_adj['dribbling']

g3 = top_players_adj['curve']

h3 = top_players_adj['long_passing']

i3 = top_players_adj['ball_control']

j3 = top_players_adj['shot_power']

k3 = top_players_adj['long_shots']

l3 = top_players_adj['interceptions']

m3 = top_players_adj['marking']

n3 = top_players_adj['standing_tackle']

o3 = top_players_adj['sliding_tackle']

fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3,5)



ax1.scatter(x,a3)

ax1.set_title('crossing')

ax2.scatter(x,b3)

ax2.set_title('finishing')

ax3.scatter(x,c3)

ax3.set_title('heading_accuracy')

ax4.scatter(x,d3)

ax4.set_title('short_passing')

ax4.set_ylim(ymin=50)

ax5.scatter(x,e3)

ax5.set_title('volleys')

ax5.set_ylim(ymin=0)

ax6.scatter(x,f3)

ax6.set_title('dribbling')

ax6.set_ylim(ymin=0)

ax7.scatter(x,g3)

ax7.set_title('curve')

ax7.set_ylim(ymin=0)

ax8.scatter(x,h3)

ax8.set_title('long_passing')

ax8.set_ylim(ymin=0)

ax9.scatter(x,i3)

ax9.set_title('ball_control')

ax9.set_ylim(ymin=40)

ax10.scatter(x,j3)

ax10.set_title('shot_power')

ax10.set_ylim(ymin=0)

ax11.scatter(x,k3)

ax11.set_title('long_shots')

ax11.set_ylim(ymin=0)

ax12.scatter(x,l3)

ax12.set_title('interceptions')

ax12.set_ylim(ymin=0)

ax13.scatter(x,m3)

ax13.set_title('marking')

ax13.set_ylim(ymin=0)

ax14.scatter(x,n3)

ax14.set_title('standing_tackle')

ax14.set_ylim(ymin=0)

ax15.scatter(x,o3)

ax15.set_title('sliding_tackle')

ax15.set_ylim(ymin=0)

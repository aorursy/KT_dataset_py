import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")  

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"     
df= pd.read_csv('/kaggle/input/ultimate-ufc-dataset/ufc-master.csv')
sns.countplot(data=df.sort_values(by='weight_class') , y = 'weight_class' , hue ='gender')

plt.xlabel('Number of Fights')

plt.ylabel('Weight Class')

plt.show();
sns.countplot(data=df , y = 'gender')

plt.ylabel('Gender')

plt.xlabel('Number of fights')

plt.show();
temp_fighter_list = pd.DataFrame(df['R_fighter'].append(df['B_fighter']).value_counts()).reset_index()

temp_fighter_list.columns = ['Name','count']

sns.barplot(data=temp_fighter_list.head(10),y = 'Name' ,x = 'count') # order = temp_fighter_list['Name'].value_counts())

plt.xlabel('Number of Fights')

plt.show();
fighter = 'Donald Cerrone'

temp_fighter_R = df[df['R_fighter'] == fighter ]

temp_fighter_R['side'] = 'Red'

temp_fighter_B = df[df['B_fighter'] == fighter ]

temp_fighter_B['side'] = 'Blue'

temp_fighter = pd.concat([temp_fighter_R,temp_fighter_B])

temp_fighter.sample(5)
def win_lose(winner, side) : # Did he win/lose

    return 'win' if winner == side else 'lose'  

temp_fighter['win_lose'] = temp_fighter.apply(lambda x : win_lose(x['Winner'],x['side']),axis = 1 )
sns.countplot(data=temp_fighter.sort_values(by='finish_round') , x = 'finish_round' , hue ='win_lose')

plt.xlabel('Round #')

plt.ylabel('Win/Lose count')

plt.show();
#finish 

sns.countplot(data=temp_fighter.sort_values(by='finish') , x = 'finish' , hue ='win_lose')

plt.xlabel('Round #')

plt.ylabel('Win/Lose count')

plt.show();
def age_during_fight(side,R_age,B_age): return R_age if side == 'Red' else B_age  # What was his age ?

def age_opponent_fight(side,R_age,B_age): return R_age if side == 'Blue' else B_age  # What was his age ?



temp_fighter['fighter_age'] = temp_fighter.apply(lambda x : age_during_fight(x['side'],x['R_age'],x['B_age']),axis = 1 )

temp_fighter['opponent_age'] = temp_fighter.apply(lambda x : age_opponent_fight(x['side'],x['R_age'],x['B_age']),axis = 1 )

temp_fighter['age_diff'] = temp_fighter['fighter_age'] - temp_fighter['opponent_age']
palette_gray = ["#95a5a6"]

sns.set()

cmap = sns.cubehelix_palette(light=1, as_cmap=True)

sns.countplot(data=temp_fighter.sort_values(by='fighter_age') , x = 'fighter_age',palette=palette_gray)

plt.show();
palette_blue_gray = ["#3498db","#95a5a6" ]

sns.countplot(data=temp_fighter.sort_values(by='fighter_age') , x = 'fighter_age' , hue = 'win_lose',palette = palette_blue_gray)

plt.show();
sns.countplot(data=temp_fighter.sort_values(by='fighter_age') , x = 'age_diff' ,palette = palette_gray)

plt.xlabel('Age difference between fighters')

plt.ylabel('Number of fights')

plt.show();
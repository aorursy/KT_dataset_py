!pip install chart_studio
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import chart_studio.plotly as py

from pylab import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



foot = pd.read_csv("../input/epldata_final.csv")



params = {'legend.fontsize': 'small',

          'figure.figsize': (15, 5),

         'axes.labelsize': 'x-large',

         'axes.titlesize':'x-large',

         'xtick.labelsize':'x-large',

         'ytick.labelsize':'x-large'}



plt.rcParams.update(params)

foot.columns
pl_under = foot[foot["age"] <= 21]



# Iterate over lang column in DataFrame

def count_u21(pl_under, *args): 



    # Initialize an empty dictionary for clubs 

    u21_dir = {}



    # Iterate over column names in args

    for col_name in args: 

    

        # Extract column from DataFrame: col

        col = pl_under[col_name]



        for entry in col:



            # If the club is in u21_count_clubs, add 1

            if entry in u21_dir.keys():

                u21_dir[entry] += 1

            # Else add the club to u21_count_clubs, set the value to 1

            else:

                u21_dir[entry] = 1



        # Print the populated dictionary

        return u21_dir



clubs_h = count_u21(pl_under,"club")



pl_e = pl_under.loc[:, ["name", "club", "age", "position", "market_value"]]

# pl_e1 = pl_e(sorted(str('market_value')))

# print(pl_e1)
clubs_p = count_u21(pl_under,"position")

print(clubs_p)
goalkeepers = clubs_p.get("GK")

defenders = clubs_p.get("CB")+clubs_p.get("LB")+clubs_p.get("RB")

midfielders =  clubs_p.get("CM")+ clubs_p.get("AM")+clubs_p.get("DM")

attackers =  clubs_p.get("CF")+clubs_p.get("LW")+clubs_p.get("RW")+clubs_p.get("SS")



sum_pos = goalkeepers + defenders + midfielders + attackers



print("Goalkeepers: " + str(goalkeepers))

print("Defenders: " + str(defenders))

print("Midfielders: " + str(midfielders))

print("Attackers: " + str(attackers))



d = dict(((k, eval(k)) for k in ('goalkeepers', 'defenders', 'midfielders', 'attackers')))



plt.bar(d.keys(), d.values(), color='b')

print(clubs_h)
plt.bar(list(clubs_h.keys()), clubs_h.values(), color='b')



plt.xlabel('Clubs')

plt.ylabel('No. of players')

plt.title("U21 distribution - Clubs")



dict_clubs = {

    'Arsenal': 'ARS', 

    'Bournemouth': 'BOU', 

    'Chelsea': 'CHE', 

    'Crystal+Palace': 'CRY',

    'Everton': 'EVE',

    'Huddersfield': 'HUD', 

    'Leicester+City': 'LEI', 

    'Liverpool': 'LIV', 'Swansea'

    'Machester+City': 'MCI',

    'Manchester+United': 'MUN', 

    'Newcastle+United': 'NEW', 

    'Southampton': 'SOU',

    'Stoke+City': 'STK', 

    'Swansea': 'SWA', 

    'Tottenham': 'TOT',

    'Watford': 'WAT', 

    'West+Brom': 'WBA', 

    'West+Ham': 'WHU',

}



plt.xticks([i for i in range(len(dict_clubs.values()))] , dict_clubs.values())



plt.show()
under_21_big6 = foot[np.logical_and(foot["age"] <= 21, foot["big_club"] == 1)]

clubs_top6 = count_u21(under_21_big6,"club")

print(clubs_top6)
plt.bar(list(clubs_top6.keys()), clubs_top6.values(), color='b')



plt.xlabel('Clubs')

plt.ylabel('No. of players')

plt.title("U21 distribution - TOP6")



plt.xticks([i for i in range(len(clubs_top6.values()))] , clubs_top6.keys() )



plt.show()
for lab, row in under_21_big6.iterrows() :   

    print(row["name"] + " play in " + row["club"])
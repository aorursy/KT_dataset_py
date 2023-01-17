# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

from PIL import Image, ImageDraw, ImageFont

from sklearn import tree

from IPython.display import Image as PImage

import matplotlib.pyplot as plt

import matplotlib

import seaborn as sns

matplotlib.style.use('fivethirtyeight')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_call

from subprocess import check_output

import re

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shots = pd.read_csv('../input/nba-shot-logs/shot_logs.csv', header=0)

print(shots.head())
closest_def = shots[["CLOSE_DEF_DIST", "SHOT_RESULT"]]

closest_def.loc[:,"SHOT_RESULT"] = closest_def.loc[:,"SHOT_RESULT"].map(lambda x: 1 if x=="made" else 0)



# The distribution is pretty big, so I grouped them into 70 bins with an equal range

closest_def["BINS"] = pd.cut(closest_def.CLOSE_DEF_DIST, 70)



# DF of bin to defender dist

defender_bin = closest_def[["BINS", "CLOSE_DEF_DIST"]]



shots_bins = closest_def[["BINS", "SHOT_RESULT"]]

# DF of bin to average shot percentage

shots_bins = shots_bins.groupby(["BINS"]).mean()



def_plot = defender_bin.plot.hist(bins=70)

shot_plot = shots_bins.plot.line()





player_shots = shots[["GAME_ID", "player_name", "SHOT_NUMBER", "SHOT_RESULT", "SHOT_DIST"]]

player_shots.loc[:, 'previous'] = np.zeros(len(shots))

player_shots.loc[:, 'dist_diff'] = np.zeros(len(shots))

for i, row in enumerate(player_shots[1:].iterrows()):

    if i>0 and player_shots.loc[i,'GAME_ID'] == player_shots.loc[i-1,'GAME_ID']:

                player_shots.loc[i,'previous'] = player_shots.loc[i-1,'SHOT_RESULT']

                player_shots.loc[i,'dist_diff'] = player_shots.loc[i,'SHOT_DIST'] - player_shots.loc[i-1,'SHOT_DIST']
# Made and missed after previous shot was made

prev_made = player_shots[player_shots.previous == "made"]



made = prev_made[prev_made.SHOT_RESULT=="made"]

missed = prev_made[prev_made.SHOT_RESULT=="missed"]



percent_made = round(len(made)/(len(made)+len(missed)), 2)

percent_missed = round(len(missed)/(len(made)+len(missed)), 2)

print (f"After making a shot, there is a {percent_made}% of making the next shot and {percent_missed}% of missing the shot.")





# Lets see offset of the next shot after making the previous one

dist_diff = prev_made.dist_diff

mean_dist_diff = dist_diff.mean()

print (f"After a make, the next shot is on average {mean_dist_diff} ft. different")

histo = dist_diff.hist(bins=30)

# Lets do the same thing as above for missed shots



# Made and missed after previous shot was made

prev_missed = player_shots[player_shots.previous == "missed"]



made = prev_missed[prev_missed.SHOT_RESULT=="made"]

missed = prev_missed[prev_missed.SHOT_RESULT=="missed"]



percent_made = round(len(made)/(len(made)+len(missed)), 2)

percent_missed = round(len(missed)/(len(made)+len(missed)), 2)

print (f"After making a shot, there is a {percent_made}% of making the next shot and {percent_missed}% of missing the shot.")





# Lets see offset of the next shot after making the previous one

dist_diff = prev_missed.dist_diff

mean_dist_diff = dist_diff.mean()

print (f"After a miss, the next shot is on average {mean_dist_diff} ft. different")

histo = dist_diff.hist(bins=30)





def get_clean_player_features(name, shots):

    player_shots = shots[shots.player_name==name]

    dropping = ['GAME_ID','MATCHUP','W','FINAL_MARGIN','SHOT_RESULT', 'CLOSEST_DEFENDER_PLAYER_ID','GAME_CLOCK','player_name','player_id','PTS','CLOSEST_DEFENDER', 'PERIOD']

    # Keep only the features used in Decision Tree

    player_shots = player_shots.drop(dropping, axis = 1)



    # # Make numerical values for this

    player_shots['LOCATION'][player_shots.LOCATION == 'H'] = 1

    player_shots['LOCATION'][player_shots.LOCATION == 'A'] = 0



    # For Shot Clock expiry



    player_shots = player_shots.fillna(0)

    return player_shots
def calc_metrics(pred, test):

    true_pos = 0

    false_pos = 0

    true_neg = 0

    false_neg = 0

    for i in range(len(pred)):

        if pred[i] == 1 and test[i] == 1:

            true_pos += 1

        elif pred[i] == 1 and test[i] == 0:

            false_pos += 1

        elif pred[i] == 0 and test[i] == 1:

            false_neg += 1

        elif pred[i] == 0 and test[i] == 0:

            true_neg += 1

            

    precision = round(true_pos / (true_pos + false_pos), 2)

    recall = round(true_pos / (true_pos + false_neg), 2)

    accuracy = round((true_pos+true_neg) / (true_pos + false_pos + true_neg + false_neg), 2)            

    

    print (f"Precision: {precision} | Recall: {recall} | Accuracy: {accuracy}")

    

    


def get_decision_tree(player_shots):

    shot_results = player_shots.FGM

    feature_set = player_shots.drop(['FGM'], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(

         feature_set, shot_results, test_size=0.20)



    decision_tree = tree.DecisionTreeClassifier(max_depth = 3, min_samples_leaf = 3)

    decision_tree.fit(x_train, y_train)

    y_pred = decision_tree.predict(x_test)

    calc_metrics(y_pred, y_test.tolist())

    

    with open("tree1.dot", 'w') as f:

        f = tree.export_graphviz(decision_tree,

                                  out_file=f,

                                  max_depth = 7,

                                  impurity = False,

                                  feature_names = x_test.columns.values,

                                  class_names = ['No', 'Yes'],

                                  rounded = True,

                                  filled= True)

        



    #Convert .dot to .png to allow display in web notebook

    check_call(['dot','-Tpng','tree1.dot','-o','tree1.png', '-Gdpi=600'])



    plt.figure(figsize = (14, 18))

    plt.imshow(plt.imread('tree1.png'))

    plt.axis('off');

    plt.show();

    

#     img = Image.open("tree1.png")

#     draw = ImageDraw.Draw(img)

#     img.save('sample-out.png')

#     PImage("sample-out.png")

player_name = ['stephen curry', 'lebron james', 'anthony davis']



for each in player_name:

    print(f"Looking at player {each}")

    player_shots = get_clean_player_features(each, shots)

    get_decision_tree(player_shots)
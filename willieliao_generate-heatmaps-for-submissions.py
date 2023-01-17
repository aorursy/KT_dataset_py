import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import re

import sys



YEAR = 2017



seeds = pd.read_csv('../input/TourneySeeds.csv')

seeds = seeds.loc[seeds.Season==YEAR]

teamIds = seeds.Team.values # team ids conveniently ordered by region, seed

teams = pd.read_csv('../input/Teams.csv')

teamsd = dict(teams[['Team_Id','Team_Name']].values)

# maps global team ids to 0..67 based on 2017 tournament region & seed

teamId2Index = {t:i for i,t in enumerate(teamIds)}

heatmapLabels = [teamsd[teamIds[i]] for i in range(len(teamIds))]

nteams = len(heatmapLabels)





# TODO a few submissions are zips within zips!

def extractSubmission(id):

    os.system("unzip -d %d ../input/predictions/%d.zip" % (id, id))    

    return os.path.join(str(id), os.listdir("%d" % id)[0])





def loadSubmission(filename):

    df = pd.read_csv(filename)

    df.columns = df.columns.str.lower()

    # t1 is always the lower team id

    df['t1'] = df.apply(lambda x: int(x.id.split('_')[1]), axis=1)

    df['t2'] = df.apply(lambda x: int(x.id.split('_')[2]), axis=1)

    df[['t1','t2']] = df[['t1','t2']].applymap(teamId2Index.get)

    # better to leave unassigned diagonal as zero, stands out more

    m = np.zeros((nteams, nteams))

    m[df.t1, df.t2] = df.pred

    m[df.t2, df.t1] = 1 - df.pred

    return m





def showHeatmap(probs, filename):

    plt.clf()

    fig, ax = plt.subplots()

    fig.set_size_inches(10, 10)

    # try different colormaps:

    # http://matplotlib.org/users/colormaps.html

    heatmap = ax.pcolormesh(probs, vmin=0, vmax=1, cmap=plt.cm.seismic)



    # put the major ticks at the middle of each cell

    ax.set_xticks(np.arange(nteams)+0.5, minor=False)

    ax.set_yticks(np.arange(nteams)+0.5, minor=False)



    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.spines['left'].set_visible(False)



    # want a more natural, table-like display

    ax.invert_yaxis()

    ax.tick_params(direction='out')

    ax.xaxis.tick_top()

    ax.yaxis.tick_left()

    plt.xticks(rotation=90)



    ax.set_xticklabels(heatmapLabels, fontsize=8)

    ax.set_yticklabels(heatmapLabels, fontsize=8)

    plt.show()

    plt.savefig(filename, bbox_inches='tight')
KAGGLERS = [ 'Willie Liao' ]



submissions = pd.read_csv('../input/team_submission_key.csv')



for i, row in submissions.ix[submissions.TeamName.isin(KAGGLERS)].iterrows():

    csv = extractSubmission(row.Id)

    probs = loadSubmission(csv)

    print(row.TeamName, '-', csv)

    showHeatmap(probs, '%d.png' % row.Id)
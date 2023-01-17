# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import collections

from os import listdir

from os.path import isfile, join

import time

import datetime

import sys

import matplotlib.pyplot as plt



class streamTrend:

    def __init__(self, noOfTopicsToTrack = 33):

        # number of items to track for heavy hitting

        self.noOfTopics = noOfTopicsToTrack

        self.topicsList = collections.Counter()



    def updateItemHit(self, data):

        if data in self.topicsList:

            self.topicsList[data] += 1

        elif len(self.topicsList) < self.noOfTopics:

            self.topicsList[data] = 1

        else:

            for topic in self.topicsList:

                self.topicsList[topic] -= 1

            # remove 0 or -ve counts

            self.topicsList += collections.Counter()



    def getTrends(self):

        return self.topicsList.most_common()



if __name__ == '__main__':

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    print ("Started calculating artist trends..", st)

    data = pd.read_csv("../input/songdata.csv",  usecols=[0,3])

    # print (data)

    mostTrending = streamTrend(50);

    data = data.fillna({'artist': 'missing'})

    #print (data)

    for index, row in data.iterrows():

        # print (row['state'])

        if row['artist'] is not '' and row['artist'] is not 'missing':

            mostTrending.updateItemHit(row['artist'])                

    trends = mostTrending.getTrends()

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    print ("Done with artist trends calculations..", st, "\treturned trends are =", trends)

    x = [1,2,3,4,5,6,7,8,9,10]

    y = []

    value = []

    gettopTen = trends[:10]

    for i, val in enumerate(gettopTen):

        y.append(val[0])

        value.append(val[1])

    plt.bar(x, value, align='center')

    plt.xticks(x, y, rotation='vertical')

    plt.xlabel('Artist', fontsize=18)

    fig = plt.gcf()



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import collections

from os import listdir

from os.path import isfile, join

import time

import datetime

import sys

import matplotlib.pyplot as plt



class streamTrend:

    def __init__(self, noOfTopicsToTrack = 33):

        # number of items to track for heavy hitting

        self.noOfTopics = noOfTopicsToTrack

        self.topicsList = collections.Counter()



    def updateItemHit(self, data):

        if data in self.topicsList:

            self.topicsList[data] += 1

        elif len(self.topicsList) < self.noOfTopics:

            self.topicsList[data] = 1

        else:

            for topic in self.topicsList:

                self.topicsList[topic] -= 1

            # remove 0 or -ve counts

            self.topicsList += collections.Counter()



    def getTrends(self):

        return self.topicsList.most_common()



if __name__ == '__main__':

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    # let's also calculate top words in lyrics

    print ("Started calculating words trends in lyrics..", st)

    mostTrendingWord = streamTrend(419);

    data = pd.read_csv("../input/songdata.csv",  usecols=[0,3])

    data = data.fillna({'text': 'missing'})

    totalWords=0

    for index, row in data.iterrows():

        if row['text'] is not '' and row['text'] is not 'missing':

            lyrics = row['text'];

            for word in lyrics.split():

                mostTrendingWord.updateItemHit(word.lower())

                totalWords+=1

    trends = mostTrendingWord.getTrends()

    ts = time.time()

    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')

    print ("Done with words trends calculations in lyrics ..", st,"\tTotal words =", totalWords, "\treturned trends are =", trends)

    x = [1,2,3,4,5,6,7,8,9,10]

    y = []

    value = []

    gettopTen = trends[:10]

    for i, val in enumerate(gettopTen):

        y.append(val[0])

        value.append(val[1])

    plt.bar(x, value, align='center')

    plt.xticks(x, y, rotation='vertical')

    plt.xlabel('Words in lyrics', fontsize=18)

    fig = plt.gcf()

    



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
from numpy.random import *

now_states=['sleep','eat','run']

incident_seq=[['ss','se','sr'],['es','ee','er'],['rs','re','rr']]

transitionMatrix=[[0.6,0.2,0.2],[0.1,0.3,0.6],[0.3,0.3,0.4]]

for line in transitionMatrix:

    if np.sum(np.array(line))!=1:

        print('There should be sum to 1')

    else:

        print('All are going on as usual!')
def markov_forecast(n_day):

    activity_today='sleep'

    activity_list=[activity_today]

    i=0

    probability=1

    while i!=n_day:

        if activity_today=='sleep':

            activ=choice(incident_seq[0],replace=True,p=transitionMatrix[0])

            if activ=='ss':

                activity_today='sleep'

                probability*=0.6

                pass

            elif activ=='se':

                activity_today='eat'

                probability*=0.2

                activity_list.append('eat')

            else:

                activity_today='run'

                probability*=0.2

                activity_list.append('run')

        elif activity_today=='eat':

            activ=choice(incident_seq[1],replace=True,p=transitionMatrix[1])

            if activ=='ee':

                activity_today='eat'

                probability*=0.3

                pass

            elif activ=='es':

                activity_today='sleep'

                probability*=0.1

                activity_list.append('sleep')

            else:

                activity_today='run'

                probability*=0.6

                activity_list.append('run')

        else:

            activ=choice(incident_seq[2],replace=True,p=transitionMatrix[2])

            if activ=='rr':

                activity_today='run'

                probability*=0.4

                pass

            elif activ=='re':

                activity_today='eat'

                probability*=0.3

                activity_list.append('eat')

            else:

                activity_today='rs'

                probability*=0.3

                activity_list.append('sleep')

        i+=1

    print(probability)

    return activity_list

    

            

            
markov_forecast(2)

new_list=[]

for i in range(1000000):

    new_list.append(markov_forecast(2))

count=0

for i in new_list:

    if len(i)==2 and i[2]=='run':

        count+=1

percentage=(count/1000000)*100

print('{0:.2f}%'.format(percentage))
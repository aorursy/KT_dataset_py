# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/kernel3eee1139a8/data.csv')

data.head()
data.columns
data.drop(['Title','Release Year','Director','Cast','Wiki Page','Origin/Ethnicity','Genre','unknown'],axis=1,inplace=True)
data.columns
from sklearn.cluster import KMeans

Xcluster = data.drop(['Plot','Summary','Cleaned'],axis=1)

kmeans = KMeans(n_clusters=3)

kmeans.fit(Xcluster)

pd.Series(kmeans.predict(Xcluster)).value_counts()
centers = kmeans.cluster_centers_

centers
data['Label'] = kmeans.labels_

Xcluster['Label'] = kmeans.labels_
for cluster in [0,1,2]:

    subset = Xcluster[Xcluster['Label']==cluster]

    subset.drop(['Label'],axis=1,inplace=True)

    indexes = subset.index

    subset = subset.reset_index().drop('index',axis=1)

    center = centers[cluster]

    scores = {'Index':[],'Distance':[]}

    

    for index in range(len(subset)):

        scores['Index'].append(indexes[index])

        scores['Distance'].append(np.linalg.norm(center-np.array(subset.loc[index])))

        

    scores = pd.DataFrame(scores)

    print('Cluster',cluster,':',scores[scores['Distance']==scores['Distance'].min()]['Index'].tolist())
data.loc[[4018,15176,9761]]['Plot']
data.loc[15176]['Summary']
data
import time

starting = []

print("Indicate if like (1) or dislike (0) the following three story snapshots.")



print("\n> > > 1 < < <")

print('On a neutral island in the Pacific called Shadow Island (above the island of Formosa), run by American gangster Lucky Kamber, both sides in World War II attempt to control the secret of element 722, which can be used to create synthetic aviation fuel.')

time.sleep(0.5) #Kaggle sometimes has a glitch with inputs

while True:

    response = input(':: ')

    try:

        if int(response) == 0 or int(response) == 1:

            starting.append(int(response))

            break

        else:

            print('Invalid input. Try again')

    except:

        print('Invalid input. Try again')





print('\n> > > 2 < < <')

print('Jake Rodgers (Cedric the Entertainer) wakes up near a dead body. Freaked out, he is picked up by Diane.')

time.sleep(0.5) #Kaggle sometimes has a glitch with inputs

while True:

    response = input(':: ')

    try:

        if int(response) == 0 or int(response) == 1:

            starting.append(int(response))

            break

        else:

            print('Invalid input. Try again')

    except:

        print('Invalid input. Try again')



print('\n> > > 3 < < <')

print("Jewel thief Jack Rhodes, a.k.a. 'Jack of Diamonds', is masterminding a heist of $30 million worth of uncut gems. He also has his eye on lovely Gillian Bromley, who becomes a part of the gang he is forming to pull off the daring robbery. However, Chief Inspector Cyril Willis from Scotland Yard is blackmailing Gillian, threatening her with prosecution on another theft if she doesn't cooperate in helping him bag the elusive Rhodes, the last jewel in his crown before the Chief Inspector formally retires from duty.")

time.sleep(0.5) #Kaggle sometimes has a glitch with inputs

while True:

    response = input(':: ')

    try:

        if int(response) == 0 or int(response) == 1:

            starting.append(int(response))

            break

        else:

            print('Invalid input. Try again')

    except:

        print('Invalid input. Try again')
X = data.loc[[9761,15176,4114]].drop(['Plot','Summary','Cleaned'],axis=1)

y = starting

data.drop([9761,15176,4114],inplace=True)
from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import shuffle



from tqdm import tqdm

subset = data.drop(['Plot','Summary','Cleaned'],axis=1)

while True:

    print("\n> > > LOADING NEW STORY...")

    dec = DecisionTreeClassifier().fit(X,y)

    

    dic = {'Index':[],'Probability':[]}

    subdf = shuffle(subset).head(5_000) #select about 1/6 of data

    for index in tqdm(subdf.index.values):

        dic['Index'].append(index)

        dic['Probability'].append(dec.predict_proba(np.array(subdf.loc[index]).reshape(1, -1))[0][1])

    

    dic = pd.DataFrame(dic)

    index = dic[dic['Probability']==dic['Probability'].max()].reset_index(drop=False).loc[0,'Index']

    

    print('> > > Would you be interested in this snippet from a story? (1/0/-1 to quit) < < <')

    print(data.loc[index]['Summary'])

    time.sleep(0.5)

    

    while True:

        response = input(':: ')

        try:

            if int(response) == 0 or int(response) == 1:

                response = int(response)

                break

            else:

                print('Invalid input. Try again')

        except:

            print('Invalid input. Try again')

    

    if response == -1:

        break

        

    X = pd.concat([X,pd.DataFrame(data.loc[index].drop(['Plot','Summary','Cleaned'])).T])

        

    if response == 0:

        y.append(0)

    else:

        print('\n> > > Printing full story. < < <')

        print(data.loc[index]['Plot'])

        time.sleep(2)

        print("\n> > > Did you enjoy this story? (1/0) < < <")

        

        while True:

            response = input(':: ')

            try:

                if int(response) == 0 or int(response) == 1:

                    response = int(response)

                    break

                else:

                    print('Invalid input. Try again')

            except:

                print('Invalid input. Try again')

        if response == 1:

            y.append(1)

        else:

            y.append(0)

    data.drop(index,inplace=True)
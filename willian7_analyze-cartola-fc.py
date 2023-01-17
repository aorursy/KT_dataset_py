#import the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#create the data frame, load the data



points_2014 = pd.read_csv('../input/2014_scouts.csv')

points_2015 = pd.read_csv('../input/2015_scouts.csv')

points_2016 = pd.read_csv('../input/2016_scouts.csv')

points_2017 = pd.read_csv('../input/2017_scouts.csv')

#address a new column with reference to the year

points_2014['year'] = '2014'

points_2015['year'] = '2015'

points_2016['year'] = '2016'

points_2017['year'] = '2017'
#concatenate the data frame

points_2014_2017 = pd.concat([points_2014,points_2015,points_2016,points_2017], sort=True)
#save to csv

points_2014_2017.to_csv('Points.csv')
#Step of treating the data, verify data that have null attributes:

plt.figure(figsize=(12,6))

sns.heatmap(points_2014_2017.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#The attributes that presented null values were: clube_id, jogos_num, mando,nota, participou, partida_id, posicao_id, rodada, rodada_id, status_id, substituido, tempo_jogado, titular

#The explanation for this is that some attributes do not exist in a few years on the top cartola, there are different ways to treat this

#problem, an example would be to replace null fields with information without affecting the dataset. Because this null data set has a large

#amount of values null and mainly they do not score criteria of players on the platform, we will initially draw them from our data-frame.

points_2014_2017.drop(['clube_id','jogos_num','mando','nota','participou','partida_id','posicao_id','rodada','rodada_id','status_id','substituido','tempo_jogado','titular'],axis=1, inplace=True)

plt.figure(figsize=(12,6))

sns.heatmap(points_2014_2017.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# To verify the correlation between the variables in the data set, seeking this with more impact on the final evaluations of the player is made olad command

corr = points_2014_2017.corr()

plt.figure(figsize=(24,12))

sns.heatmap(corr,annot=True)
corr['pontos_num']
#Cartola fc scoring system:



# Ball RB (RB) + 1.5

# Falta committed (FC) - 0,5

# Gol vs. (GC) - 5.0

# Yellow Card (CA) - 2.0

# Red Card (CV) - 5.0

# Game without conceding a goal (SG) - exclusive # for the positions of goalkeeper, defender # and lateral + 5.0

# Defense difficult (DD) - exclusive to goalkeeper position + 3.0

# Defense penalty (DP) - exclusive to goalkeeper position + 7.0

# Gol suffered (GS) - exclusive for goalkeeper position - 2.0

# Missed (FS) + 0.5

# Wrong Pass (PE) - 0.3

# Assistance (A) + 5.0

# Finish on beam (FT) + 3.0

# Defended Finish (FD) + 1.2

# Finishing out (FF) + 0.8

# Gol (G) + 8.0

# Impairment (I) - 0.5

# Lost Penalty (PP) - 4.0







# From the table it is possible to notice initially that there is no correlation of the variable pontos_num with the other variables, the attributes of points that presented

# best values were: A (assist), G (goal), RB (stolen ball) and SG (play without a goal).

# In relation to these variables, those with high values were:

#A - FD (Finishing defended) and FS (Foul suffered) over 60%

#G - FD and FS and I above 60%

#RB - FC (Foul committed), FF (Finished out), FS, PE, above 60%

#SG - CA and PE above 45%



# the result of the correlation made sense in relation to the platform system.

# attention is drawn to the high correlation between handicap and goal, as well as stolen ball and finalizations.









partida = pd.read_csv('../input/2014_partidas.csv')

partida.info()
plt.figure(figsize=(12,6))

sns.heatmap(points_2014.isnull(),yticklabels=False,cbar=False,cmap='viridis')
position_2014 = pd.DataFrame(points_2014,columns=['posicao_id','pontos_num'])

position_2017 = pd.DataFrame(points_2017,columns=['posicao_id','pontos_num'])

position_2014_2017 = pd.concat([position_2014,position_2017], sort=True)

position_2014_2017.isnull().sum()

sns.heatmap(position_2014_2017.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Delete null data

position_2014_2017 = position_2014_2017.dropna(subset=['posicao_id'])

#Score by position from 2014 to 2017

position_2014_2017 = position_2014_2017.merge(pd.read_csv('../input/posicoes.csv'), how='left', left_on='posicao_id', right_on='id', suffixes=['', '_posicao'])

plt.figure(figsize=(20, 5))

plt.title("Score by position from 2014 to 2017")

position_2014_2017.groupby('nome').sum()['pontos_num'].plot.bar()

#Goleiro - Goalkeeper

#Atacante - Forward

#Lateral - full-back or right-back

#Meia - Midfield

#Zagueiro - Defender

#Técnico - Coach
pon = pd.DataFrame(points_2014, columns=['mando','pontos_num'])

#Changing the Mando attribute (0,1) to 'Casa' and 'Fora' for this was created function

#Casa - In the house of the team

#Fora - Outside the team house

def mand(mando):

    if mando == 1:

        return 'Casa'

    else:

        return 'Fora'

       





pon['nome'] = pon['mando'].apply(lambda x: mand(x))
plt.title("Points by location of games - 2014")

pon.groupby('nome').sum()['pontos_num'].plot.bar()
pon_jog = pd.DataFrame(points_2014, columns=['mando','pontos_num','posicao_id'])

pon_jog = pon_jog.merge(pd.read_csv('../input/posicoes.csv'), how='left', left_on='posicao_id', right_on='id', suffixes=['', '_posicao'])

pon_jog['cond'] = pon_jog['mando'].apply(lambda x: mand(x))

dt = pon_jog.groupby(['cond','nome']).sum().unstack('cond')['pontos_num'].plot.bar()

plt.title("Points by position and location of matches - 2014")

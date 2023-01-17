# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from time import sleep
from tqdm import tqdm
import pickle
import networkx as nx
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from ast import literal_eval
df2 = pd.read_csv('/kaggle/input/league-of-legendslol-ranked-games-2020-ver1/match_data_version1.csv')
df3 = pd.read_csv('../input/league-of-legendslol-champion-and-item-2020/riot_champion.csv')
champ_dict = dict(zip(df3.key,df3.id))
ilimit = len(df2)
play_aid = np.empty(ilimit*10,dtype='object')
play_role = np.empty(ilimit*10,dtype='object')
play_champ = np.empty(ilimit*10,dtype=np.int)

for i in tqdm(range(ilimit)):
    a=literal_eval(df2.participantIdentities[i])
    b=literal_eval(df2.participants[i])
    for ii in range(len(b)):
        role=b[ii]['timeline']['lane']
        champ = b[ii]['championId']
        pid = b[ii]['timeline']['participantId']
        for iii in range(10):
            if a[iii]['participantId'] == pid:
                aid = a[iii]['player']['currentAccountId']
                break
        play_aid[i*10+ii] = aid
        play_role[i*10+ii] = role
        play_champ[i*10+ii] = champ
df_play = pd.DataFrame({'accountId':play_aid,'role':play_role,'champion':play_champ})
df_play.to_pickle('/kaggle/working/play.pkl')
class champs:
    def __init__(self):
        self.top = None
        self.mid = None
        self.jungle = None
        self.bot = None
        self.sup = None

class players:
    def __init__(self):
        self.top = None
        self.mid = None
        self.jungle = None
        self.bot = None
        self.sup = None

class networks:
    def __init__(self):
        self.top = None
        self.mid = None
        self.jungle = None
        self.bot = None
        self.sup = None

champs.top =df_play[df_play.role=='TOP'].champion.unique()
champsum = np.zeros(len(champs.top),dtype=np.int)
players.top =df_play[df_play.role=='TOP'].accountId.unique()
ntop = len(champs.top)
networks.top= np.zeros((ntop,ntop),dtype=np.float64)
temp_nt = np.zeros((ntop,ntop),dtype=np.float64)
dft = df_play[df_play.role == 'TOP']
for player in tqdm(players.top):
    champsum = np.zeros(ntop,dtype=np.int)
    temp_nt = np.zeros((ntop,ntop),dtype=np.float64)
    dft = dft[dft.accountId == player]
    for i in range(ntop):
        champ = champs.top[i]
        champsum[i] = sum(dft.champion == champ)
    for i in range(ntop):
        for ii in range(i+1,ntop,1):
            temp_nt[ii,i] = math.sqrt(champsum[i]*champsum[ii])
    norm_factor = sum(sum(temp_nt))
    if norm_factor != 0:
        temp_nt = temp_nt/norm_factor
        networks.top += temp_nt

filename = '/kaggle/working/network_top.pkls'
outfile = open(filename,'wb')
pickle.dump(networks.top,outfile)
outfile.close()
df_cor_list=[]
weightlist=np.zeros(ntop)
for ii in range(ntop):
    weights = np.empty(ntop-1,dtype=np.float64)
    for i in range(ii):
        weights[i] = networks.top[ii,i]
    for i in range(ii+1,ntop,1):
        weights[i-1] = networks.top[i,ii]
    champlist =np.append(champs.top[:ii],champs.top[ii+1:])
    df_cor = pd.DataFrame({'champion':champlist,'weight':weights})
    df_cor_list.append(df_cor)
    if any(df_cor.weight):
        df_rec = df_cor[df_cor.weight != 0].sort_values('weight',ascending=False).reset_index(drop=True)
        if len(df_rec) != 0:
            print('{} is correlated to:'.format(champ_dict[champs.top[ii]]))
            wl=[]
            for iii in range(min(len(df_rec),3)):
                print('    {} with weight of {}'.format(champ_dict[df_rec.champion[iii]],df_rec.weight[iii]))
                wl.append(df_rec.weight[iii])
            weightlist[ii]=np.mean(wl)

print()
df_mean_weight = pd.DataFrame({'champion':champs.top,'meanweight':weightlist}).sort_values('meanweight',ascending=False).reset_index(drop=True)

for ii in range(ntop):
    if df_mean_weight.meanweight[ii] != 0:
        print('{} has correlation of {} to other champions'.format(champ_dict[df_mean_weight.champion[ii]],df_mean_weight.meanweight[ii]))
G = nx.Graph()
for i,champion in enumerate(df_mean_weight.champion):
    if df_mean_weight.meanweight[i] != 0:
        G.add_node(champ_dict[champion])

for row in range(ntop):
    for col in range(ntop):
        if networks.top[row,col] != 0:
            c1 = champs.top[row]
            c2 = champs.top[col]
            
            G.add_edge(champ_dict[c1],champ_dict[c2],weight=networks.top[row,col])
        
        
        
plt.figure(1,figsize=(40,12))      
plt.subplot(121)

nx.draw(G, with_labels=True)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

from IPython.display import display



from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





pkmn = pd.read_csv("../input/Pokemon.csv")

dex = 721
#####################         What type differences are based on primary or secondary type?

##############   that is, what difference do you get if type 1 is fighting vs if type 2 is fighting



val = pkmn['Type 1'].unique()



cnt1 = cnt2 = 0

ind1 = ind2 = 0



avg1 = [0]*len(val)

avg2 = [0]*len(val)

cnt1 = [0]*len(val)

cnt2 = [0]*len(val)

for ind, mytype in enumerate(val):

    

    for k in list(range(1,len(pkmn['#']))):

        if pkmn['Type 1'][k] == mytype:

            cnt1[ind] += 1

            avg1[ind] += pkmn["Total"][k]

                                

        

        elif pkmn['Type 2'][k] == mytype:

            cnt2[ind] += 1

            avg2[ind] += pkmn["Total"][k]

           

diff = [0]*len(val)



for i in range(0,len(val)):

    avg1[i]/=cnt1[i]

    avg2[i]/=cnt2[i]

    diff[i] = avg1[i] - avg2[i]





type1v2 = pd.DataFrame({ 'Type': val,

						 'First-type count': cnt1,

						 'Average Total Type1': avg1,

						 'Second-type count': cnt2,

						 'Average Total Type2': avg2,

						 'Difference': diff



	})

    

display(type1v2)
off_stats = [0]*len(val)

def_stats = [0]*len(val)

for ind, mytype in enumerate(val):

    for k in list(range(1,len(pkmn['#']))):

        if pkmn['Type 1'][k] == mytype or pkmn['Type 2'][k] == mytype:

            def_stats[ind] += pkmn['HP'][k] + pkmn['Defense'][k] + pkmn['Sp. Def'][k]

            off_stats[ind] += pkmn['Attack'][k] + pkmn['Sp. Atk'][k] + pkmn['Speed'][k]

            



for x in range(0, len(val)):

    off_stats[x] /= (cnt1[x]+cnt2[x])

    def_stats[x] /= (cnt1[x]+cnt2[x])

    

offVsdef = pd.DataFrame({ 'Type': val,

						 'Offensive Stat Avg.': off_stats,

						 'Defensive Stat Avg': def_stats

						 

	})

    

display(offVsdef)

    
legend_pkmn = pkmn[pkmn['Legendary']==True]



legend_avg = 0

for x in list(range(1,len(pkmn['#']))):

	if pkmn['Legendary'][x] == True:

		legend_avg +=pkmn['Total'][x]

legend_avg /= len(legend_pkmn['#'])

print(legend_avg)
pseudo_legendary = pkmn[(pkmn['Legendary']==False)&(pkmn['Total']>=0.9*legend_avg)]



print(pseudo_legendary.shape)



for y in range(1,len(pseudo_legendary['Total'])):

	k = pseudo_legendary.iloc[y]['Name']

	print(k)
for y in range(1,len(pseudo_legendary['Total'])):

	k = pseudo_legendary.iloc[y]['Name']

	if "Mega" not in k:

		print(k)
del pkmn['Total'], pkmn['#'], pkmn['Legendary']
sns.set_style("whitegrid")

with sns.color_palette([

    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    plt.figure(figsize=(8,8))

    plt.ylim(0, 275)

    pkmn_thisgen = pkmn[(pkmn['Generation']==1)|(pkmn['Generation']==2)|(pkmn['Generation']==3)]

    pkmn_secondsetgen = pkmn[(pkmn['Generation']==4)|(pkmn['Generation']==5)|(pkmn['Generation']==6)]

    sns.boxplot(data=pkmn_thisgen)

    sns.plt.show()

    plt.figure(figsize=(8,8))

    plt.ylim(0, 275)

    sns.boxplot(data=pkmn_secondsetgen)

    sns.plt.show()
pkmn_ourgen = pkmn[pkmn["Generation"]==1]

with sns.color_palette([

    "#8ED752", "#F95643", "#53AFFE", "#C3D221", "#BBBDAF",

    "#AD5CA2", "#F8E64E", "#F0CA42", "#F9AEFE", "#A35449",

    "#FB61B4", "#CDBD72", "#7673DA", "#66EBFF", "#8B76FF",

    "#8E6856", "#C3C1D7", "#75A4F9"], n_colors=18, desat=.9):

    plt.figure(figsize=(6,6))

    plt.ylim(0, 275)

    for i in range(1,7):

        pkmn_thisgen = pkmn[pkmn['Generation']==i]

        sns.boxplot(data=pkmn_thisgen)

        sns.plt.title('Gen %d' %i)

        sns.plt.show()
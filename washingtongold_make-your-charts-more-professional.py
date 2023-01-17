sns.lineplot([1,2,3,4,5,6,7],

             [3,6,4,0,-3,-1,0])

sns.lineplot([1,2,3,4,5,6,7],

             [3.5,2.3,-2.3,-1,2,3,4])
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

plt.figure(figsize=(6,4))

fig,ax = plt.subplots()

sns.lineplot([1,2,3,4,5,6,7],

             [3,6,4,0,-3,-1,0],

             marker='o',color='#4285F4',alpha=1)

sns.lineplot([1,2,3,4,5,6,7],

             [3.5,2.3,-2.3,-1,2,3,4],

             marker='o',color='#EA4335',alpha=1)

sns.lineplot([1,7],[0,0],color='black',alpha=0.5)

ax.text(7.5, 3.7, 'Red Line', fontsize=15, color='dimgrey')

ax.text(7.5, -0.3, 'Blue Line', fontsize=15, color='dimgrey')

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(6,4))

fig,ax = plt.subplots()

plt.fill_between([1,2,3,4,5,6,7],

                 [3,6,4,0,-3,-1,0], color="#4285F4", alpha=0.2)

plt.fill_between([1,2,3,4,5,6,7],

                 [3.5,2.3,-2.3,-1,2,3,4], color='#EA4335', alpha=0.2)

sns.lineplot([1,2,3,4,5,6,7],

             [3,6,4,0,-3,-1,0],

             marker='o',color='#4285F4',alpha=1)

sns.lineplot([1,2,3,4,5,6,7],

             [3.5,2.3,-2.3,-1,2,3,4],

             marker='o',color='#EA4335',alpha=1)

sns.lineplot([1,7],[0,0],color='black',alpha=0.5)

ax.text(7.5, 3.7, 'Red Line', fontsize=15, color='dimgrey')

ax.text(7.5, -0.3, 'Blue Line', fontsize=15, color='dimgrey')

plt.show()


plt.figure(figsize=(6,4))

sns.lineplot([1,2,3,4,5,6,7],

             [3,6,4,0,-3,-1,0],

             marker='o',color='#4285F4')#,alpha=1)

sns.lineplot([1,2,3,4,5,6,7],

             [3.5,2.3,-2.3,-1,2,3,4],

             marker='o',color='#EA4335')#,alpha=0.5)

sns.lineplot([1,7],[0,0],color='black',alpha=0.5)
sns.set_style('white')

sns.barplot(['A','B','C','D'],[13,16,6,14])
fig, ax = plt.subplots()

sns.barplot(['A','B','C','D'],[13,16,6,14],palette=['#4285F4','#EA4335','#FBBC05','#34A853'],alpha=0.85)

sns.set_style('whitegrid')

#sns.barplot(['A','B','C','D'],[13,16,6,14])

totals = []



# find the values and append to list

for i in ax.patches:

    totals.append(i.get_height())



# set individual bar lables using above list

total = sum(totals)



# set individual bar lables using above list

for i in ax.patches:

    # get_x pulls left or right; get_height pushes up or down

    ax.text(i.get_x()+0.3, i.get_height()+.5, \

            str(int(i.get_height())), fontsize=15,

                color='dimgrey')
plt.figure(figsize=(4,4))

x, values = ['A','B','C','D'], [13,16,6,14]

# stem function: first way

plt.stem(x, values)

plt.ylim(0, 18)

#plt.show()

plt.stem(values)

#plt.show()

 

#stem function: second way

(markerline, stemlines, baseline) = plt.stem(x, values)

plt.setp(baseline, visible=False)

#plt.show()
df = sns.load_dataset('tips')

sns.boxplot(x="day", y="total_bill",data=df)
import seaborn as sns

df = sns.load_dataset('tips')

plt.figure(figsize=(5,5))

# Grouped violinplot

sns.violinplot(x="day", y="total_bill", data=df,palette=['#99ffcc','#b3ffb3','#ffd9b3','#ff9999'],alpha=1)

sns.lineplot([0,1],[df[df['day']=='Thur'].median()['total_bill'],

                    df[df['day']=='Fri'].median()['total_bill']],color='black',alpha=0.7)

sns.lineplot([1,2],[df[df['day']=='Fri'].median()['total_bill'],

                    df[df['day']=='Sat'].median()['total_bill']],color='black',alpha=0.7)

sns.lineplot([2,3],[df[df['day']=='Sat'].median()['total_bill'],

                    df[df['day']=='Sun'].median()['total_bill']],color='black',alpha=0.7)

plt.xlabel(' ')

plt.ylabel(' ')

plt.show()
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

sns.set_style('whitegrid')

df = sns.load_dataset('iris')

x=[np.random.normal() for i in range(500)]

y=[np.random.normal()*3 for i in range(500)]

# Change shape of marker

sns.scatterplot(x=x,y=y)
z =[np.random.normal()*3 for i in range(500)]
plt.figure(figsize=(5,5))

sns.kdeplot(x,y)
sns.dogplot()
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

plt.figure(figsize=(15,5))

sns.set_style('whitegrid')

g = sns.factorplot(np.array([1,2,3,4,5,6,7]), np.array([3,6,8,7,4,-1,0]), scale=.5)

lw = g.ax.lines[0].get_linewidth() # lw of first line

plt.setp(g.ax.lines,linewidth=lw)  # set lw for all lines of g axes



plt.show()
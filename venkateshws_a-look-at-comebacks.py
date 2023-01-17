# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import re

import os
path = '../input/'



os.chdir(path)

filenames =  os.listdir(path)

df = pd.DataFrame()



for filename in sorted(filenames)[1:]:

    try:

	    read_filename = path + '/' + filename

	    temp = pd.read_csv(read_filename,encoding='utf8')

	    frame = [df,temp]

	    df = pd.concat(frame)

    except UnicodeDecodeError:

        pass
df['Set_1'], df['Set_2'], df['Set_3'] = df['score'].str.split(' ',2).str
comeback = 0

for item,row in df.iterrows():

	if 'R' not in str(row['Set_2']):

		if 'R' not in str(row['Set_3']) and str(row['Set_3']) != 'nan' and 'u' not in str(row['Set_3']) and str(row['Set_3']) != '6-0 6-1' and 'D' not in str(row['Set_3']):

			set_score_Set_2 = re.sub("\(\d+\)"," ",row['Set_2'])

			set_score_Set_3 = re.sub("\(\d+\)"," ",row['Set_3'])

			Set_3 = float(set_score_Set_3.split('-')[0]) - float(set_score_Set_3.split('-')[1])

			Set_2 = float(set_score_Set_2.split('-')[0]) - float(set_score_Set_2.split('-')[1])

			if Set_3 * Set_2 > 0:

				comeback += 1



print ('Comeback %% = %f'%(100*float(comeback)/float(len(df))))
surface_group = df.groupby('surface')

y = []

x = []

for key in surface_group.groups.keys():

	comeback = 0

	count_grass = 0

	x.append(key)

	for index,row in surface_group.get_group(key).iterrows():

			count_grass += 1

			if 'R' not in str(row['Set_2']):

				if 'R' not in str(row['Set_3']) and str(row['Set_3']) != 'nan' and 'u' not in str(row['Set_3']) and str(row['Set_3']) != '6-0 6-1' and 'D' not in str(row['Set_3']):

					set_score_Set_2 = re.sub("\(\d+\)"," ",row['Set_2'])

					set_score_Set_3 = re.sub("\(\d+\)"," ",row['Set_3'])

					Set_3 = float(set_score_Set_3.split('-')[0]) - float(set_score_Set_3.split('-')[1])

					Set_2 = float(set_score_Set_2.split('-')[0]) - float(set_score_Set_2.split('-')[1])

					if Set_3 * Set_2 > 0:

						comeback += 1

	y.append(100*(float(comeback)/float(count_grass)))



x_pos = np.arange(len(x))

plt.figure(1)

plt.bar(x_pos, y, color = ['Red','Orange','Green','Blue'],align = 'center')

plt.xticks(x_pos,x)

plt.xlabel('Surface')

plt.ylabel('Comeback Percentage')
player_group = df.groupby('winner_name')



k = 0

for key in player_group.groups.keys():

	k += len(player_group.get_group(key))



avg_wins = k/len(player_group)

print ('Avg_wins per player = %f'%(float(avg_wins)))







#Who makes the most comebacks?

X = []

for key in player_group.groups.keys():

	no_of_wins = len(player_group.get_group(key))

	if no_of_wins > avg_wins:

		comeback = 0

		count_grass = 0

		# x.append(key)

		for index,row in player_group.get_group(key).iterrows():

				count_grass += 1

				if 'R' not in str(row['Set_2']):

					if 'R' not in str(row['Set_3']) and str(row['Set_3']) != 'nan' and 'u' not in str(row['Set_3']) and str(row['Set_3']) != '6-0 6-1' and 'D' not in str(row['Set_3']):

						set_score_Set_2 = re.sub("\(\d+\)"," ",row['Set_2'])

						set_score_Set_3 = re.sub("\(\d+\)"," ",row['Set_3'])

						Set_3 = float(set_score_Set_3.split('-')[0]) - float(set_score_Set_3.split('-')[1])

						Set_2 = float(set_score_Set_2.split('-')[0]) - float(set_score_Set_2.split('-')[1])

						if Set_3 * Set_2 > 0:

							comeback += 1

		X.append([key,100*(float(comeback)/float(count_grass)),no_of_wins])



X.sort(key = lambda x:(-x[1],-x[2]))

print (X)
x = []

y = []

for i,j in enumerate(X):

	x.append(j[1])

	y.append(j[2])





plt.figure(2)

plt.hist(x)

plt.xlabel('Comeback %')

plt.ylabel('Occurences')



plt.figure(3)

plt.scatter(y,x)

plt.xlabel('# wins')

plt.ylabel('Comeback %')



plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





# Any results you write to the current directory are saved as output.



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



data = pd.read_csv('../input/h1b_kaggle.csv')



#To get state information alone

data.loc[:,'WORKSITE'] = data.loc[:,'WORKSITE'].apply(lambda x:x.split(',')[-1][1:])



#MAX WAGE and Job_Title in Each State



state_group = data.groupby('WORKSITE')

for key in state_group.groups.keys():

	print(max(state_group.get_group(key)['PREVAILING_WAGE']),state_group.get_group(key).loc[state_group.get_group(key)['PREVAILING_WAGE'].idxmax()]['JOB_TITLE'],key) 

	





#How much do DATA SCIENTISTS WAGE DIFFER ACROSS STATES



job_title_group = data.groupby('JOB_TITLE')

for key in job_title_group.groups.keys():

	if key == 'DATA SCIENTIST':

		d = job_title_group.get_group(key)[['WORKSITE','PREVAILING_WAGE']]



d.boxplot(column = 'PREVAILING_WAGE', by = 'WORKSITE' , fontsize = 6)

plt.xticks(rotation = 45)

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data=pd.read_csv('../input/TimeUse.csv')

data_list = data.T.values.tolist()



household = data_list[14]

country = data_list[1][0:14]

male_household = []

female_household = []

ratio_household = []

for i in range(14):

    for j in range(2):

        s= household[i+j*14].split(':')

        duration = int(s[0])*60 + int(s[1])

        if j == 0:

            male_household.append(duration)

        else:

            female_household.append(duration)

            ratio_household.append(duration/male_household[i])





import matplotlib.pyplot as plt



y_pos = np.arange(len(country))

plt.barh(y_pos, male_household, align='center', alpha=0.5, left = 0)

plt.yticks(y_pos, country)

plt.xlabel('Minutes spent on household and family care')

plt.title('Time spent on household by MALE')

plt.show()



plt.barh(y_pos, female_household, align='center', alpha=0.5, left = 0)

plt.yticks(y_pos, country)

plt.xlabel('Minutes spent on household and family care')

plt.title('Time spent on household by FEMALE')

plt.show()



plt.barh(y_pos, ratio_household, align='center', alpha=0.5, left = 0)

plt.yticks(y_pos, country)

plt.xlabel('Minutes spent on household and family care')

plt.title('Time Ratio Female/Male spent on household')

plt.show()



print("Results show that the female/male ratio of time spent on household averages at about 1.5-2.0, while Italy and Spain have ratios over 3.0.")

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def calc_day(days):

    week_days = []

    for el in days.split(' '):

        try:

            week_days.append((int(el) - 1) % 7)

        except:

            pass

    day_visits = np.zeros(7, dtype=np.float64)

    for day in week_days:

        day_visits[day] += 1

    v_len = len(week_days)

    p_list = day_visits / v_len

    ans = [0 for i in range(7)]

    for i in range(7):

        res = np.prod(1 - p_list[:i])

        res *= p_list[i]

        ans[i] = res



    return np.array(ans).argmax() + 1

     

    
data = pd.read_csv('./dayofweek/train.csv')



res = pd.DataFrame()



res['id'] = data['id']

res['nextvisit'] = data['visits'].apply(calc_day)

res['nextvisit'] = res['nextvisit'].apply(lambda v: ' {}'.format(v))



res.to_csv('res.csv', index=False)

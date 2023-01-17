import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv('../input/HR_comma_sep.csv');

print(data['salary'].unique())

data_left = data[data['left']==1];

data_unleft = data[data['left']==0];

data_left.describe()

data_unleft.describe()

data_left_promotion = data_left[data_left['promotion_last_5years']==1];

data_left_unpromotion = data_left[data_left['promotion_last_5years']==0];

print(data_left_promotion.describe())

print(data_left_unpromotion.describe())

#data['satisfaction_level'] = (data['satisfaction_level']-np.mean(data['satisfaction_level']))/np.std(data['satisfaction_level']);

#data['last_evaluation'] = (data['last_evaluation']-np.mean(data['last_evaluation']))/np.std(data['last_evaluation']);

#data['number_project'] = (data['number_project']-np.mean(data['number_project']))/np.std(data['number_project']);

#data['average_montly_hours'] = (data['average_montly_hours']-np.mean(data['average_montly_hours']))/np.std(data['average_montly_hours']);

#data['time_spend_company'] = (data['time_spend_company']-np.mean(data['time_spend_company']))/np.std(data['time_spend_company']);

#figure = plt.figure(figsize=(5,6))

#axe = figure.add_subplot(1,1,1)

#axe.scatter(data['satisfaction_level'],data['last_evaluation']);

#axe.set_xlabel('satisfaction_level')

#axe.set_ylabel('last_evaluation')

#axe.set_title('satisfaction_level and last_evaluation')

#plt.show();

sns.PairGrid(data,hue='left',size=2).map_diag(sns.kdeplot).map_offdiag(plt.scatter);
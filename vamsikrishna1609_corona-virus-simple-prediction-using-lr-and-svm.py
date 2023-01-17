import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

import pandas as pd
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


cov = pd.read_csv('../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200206.csv'

                  , header=0

                  , names=['state','country','last_update','confirmed','suspected','recovered','death'])

cov.head()


cov['last_update'] = pd.to_datetime(cov['last_update']).dt.date

cov.info() # seeing dataset structure
# replacing state missing values by "unknow"

cov['state'] = cov['state'].fillna('unknow')



# replacing numerical variables missing values by 0

cov = cov.fillna(0)
cov

# taking cases and dates in china

china_cases_grow = cov[['last_update','confirmed','suspected','recovered','death']][cov['country']=='Mainland China']



# creating a new subset with cases over the days

china_confirmed_grow = china_cases_grow[['confirmed']].groupby(cov['last_update']).max()

china_suspected_grow = china_cases_grow[['suspected']].groupby(cov['last_update']).max()

china_recovered_grow = china_cases_grow[['recovered']].groupby(cov['last_update']).max()

china_death_grow = china_cases_grow[['death']].groupby(cov['last_update']).max()
# defyning plotsize

plt.figure(figsize=(20,10))



# creating the plot

sns.lineplot(x = china_death_grow.index

        , y = 'death'

        , color = '#4b8bbe'

        , label = 'death'

        , marker = 'o'

        , data = china_death_grow)



# titles parameters

plt.title('Growth of death rate cases in China',size=30)

plt.ylabel('Cases',size=20)

plt.xlabel('Updates',size=20)

plt.xticks(rotation=45,size=15)

plt.yticks(size=15)



# legend parameters

plt.legend(loc = "upper left"

           , frameon = True

           , fontsize = 15

           , ncol = 1

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1);
times = np.arange(1,17)

var=china_death_grow.to_numpy()

var
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

reg = LinearRegression().fit(times.reshape(-1,1), var)

reg.score(times.reshape(-1,1),var)
day=np.array([16])

valr=reg.predict(day.reshape(-1,1))

print(f'Prediction of deaths using linear regression on {int(day)}th day is {int(valr)}')
clf = SVR(degree=5,C=1000)

vals=clf.fit(times.reshape(-1,1),var).predict(times.reshape(-1,1))

plt.plot(times.reshape(-1,1),vals)

plt.scatter(times.reshape(-1,1),var)

plt.show()
value = clf.predict(np.arange(1,31).reshape(-1,1))

plt.plot(np.arange(1,31),value)

plt.scatter(np.arange(1,31),value)

plt.show()

#print(f'Prediction of deaths using SVR on {int(day)} is {int(value)}')
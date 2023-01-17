# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from datetime import date

from datetime import timedelta

import matplotlib.pyplot as plt

import gc
train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

train['Province_State']=train['Province_State'].fillna('Unknown')

def keepmonthday(x):

    x_seg=x.split('-')

    return x_seg[1]+'-'+x_seg[2]

train['Date']=train['Date'].apply(lambda x: keepmonthday(x))
compre_df = pd.read_csv("../input/countryinfo/covid19countryinfo.csv")

compre_df['region']=compre_df['region'].fillna('Unknown')
nanidex_compre=compre_df[['quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].isna().all(axis=1)
nonna_compre_df=compre_df.loc[~nanidex_compre,['region', 'country','quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].reset_index(drop=True)
train=train.merge(nonna_compre_df, how='left', left_on=['Country_Region','Province_State'], right_on=['country','region'])

train.drop(['country','region'], axis=1, inplace=True)
nanidx_train=train[['quarantine', 'schools','publicplace', 'gatheringlimit', 'gathering', 'nonessential']].isna().all(axis=1)
nonna_train=train.loc[~nanidx_train,:].reset_index(drop=True)
compare_cols=['quarantine', 'schools','publicplace', 'gathering', 'nonessential']
nonna_train['measures']=nonna_train[compare_cols].notna().sum(axis=1)
def keepmonthday2(x):

    if x is np.nan:

        return x

    else:

        x_seg=x.split('/')

        return format(int(x_seg[0]), '02')+'-'+format(int(x_seg[1]), '02')
for c in compare_cols:

    nonna_train[c]=nonna_train[c].apply(lambda x: keepmonthday2(x))
nonna_train_group=nonna_train.groupby(['Country_Region','Province_State'])
import matplotlib.transforms as mtransforms

fig_row=2

numfigs=len(nonna_train_group)

fig,ax=plt.subplots(int(np.ceil(numfigs/fig_row)),fig_row,figsize=(18,int(np.ceil(numfigs/fig_row))*(18/fig_row)))

legendlist=[]

for i,agroup in enumerate(nonna_train_group):

    #print(agroup[0])

    ridx=int(i//fig_row)

    cidx=int(i%fig_row)

    mycolumns=agroup[1][['Date', 'ConfirmedCases']]

    aline=mycolumns.set_index('Date')

    ax[ridx,cidx].plot(aline)

    row_interest=agroup[1].iloc[0,:]

    for c in compare_cols:

        xvalue=row_interest[c]

        if xvalue is np.nan:

            yvalue=np.nan

        else:

            yvalue=mycolumns.loc[mycolumns['Date']==xvalue,'ConfirmedCases']

        ax[ridx,cidx].plot(xvalue,yvalue,marker='o', markersize=8, label=c)

        if c=='gathering':

            txt=row_interest['gatheringlimit']

            if txt is np.nan:

                print(txt is np.nan)

                txt=str(int(txt))

            trans_offset = mtransforms.offset_copy(ax[ridx,cidx].transData, fig=fig,

                                       x=-0.2, y=0.10, units='inches')

            ax[ridx,cidx].text(xvalue,yvalue,txt, transform=trans_offset)

    plt.setp(ax[ridx,cidx].xaxis.get_majorticklabels(), rotation=45)

    ax[ridx,cidx].set_title('{}_{}'.format(agroup[0][0],agroup[0][1]))

    ax[ridx,cidx].set_xlabel('Date')

    ax[ridx,cidx].set_ylabel('ConfirmedCases')

    ax[ridx,cidx].legend(loc='upper left')

fig.tight_layout() 

fig.show()
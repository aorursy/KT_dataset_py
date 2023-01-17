# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


# get titanic & test csv files as a DataFrame

union_df = pd.read_csv("../input/unions_states.csv" )

union_df.head()
union_df.describe()
union_df.groupby(['Year']).groups.keys()

groupedYearUnion = union_df.groupby(['Year'])['Employment'].sum()

groupedYearUnion_df= pd.DataFrame(groupedYearUnion)

groupedYearUnion_df['Year'] = groupedYearUnion_df.index



groupedStateUnion = union_df.groupby(['State'])['Employment'].sum()

groupedStateUnion_df= pd.DataFrame(groupedStateUnion)

groupedStateUnion_df['State'] = groupedStateUnion_df.index



groupedSectorUnion = union_df.groupby(['Sector'])['Employment'].sum()

groupedSectorUnion_df= pd.DataFrame(groupedSectorUnion)

groupedSectorUnion_df['Sector'] = groupedSectorUnion_df.index



groupedPctCovUnion = union_df.groupby(['Year'])['PctCov'].sum()

groupedPctCovUnion_df= pd.DataFrame(groupedPctCovUnion)

groupedPctCovUnion_df['Year'] = groupedPctCovUnion_df.index
#ax = sns.tsplot( time="Year",data=groupedYearUnion_df)

# TimeSeries Total Employment by Year

ax = sns.regplot(x=groupedYearUnion_df['Year'], y=groupedYearUnion_df['Employment'])
# TimeSeries Total Employment by State

ax = sns.barplot(x="Employment", y="State", data=groupedStateUnion_df)
# TimeSeries Total Employment by Sector

ax = sns.barplot(x="Employment", y="Sector", data=groupedSectorUnion_df)
#ax = sns.tsplot( time="Year",data=groupedYearUnion_df)

# TimeSeries Coverage Percentage,which has sharp decrease

ax = sns.regplot(x=groupedPctCovUnion_df['Year'], y=groupedPctCovUnion_df['PctCov'])
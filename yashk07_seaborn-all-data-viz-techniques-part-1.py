# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

tips = sns.load_dataset('tips')
tips
sns.distplot(tips['total_bill']);
sns.distplot(tips['total_bill'],kde=False,bins=30); #removing the kde layer and just having histogram use.
sns.jointplot(x='total_bill',y='tip',data = tips,kind = 'scatter');
sns.jointplot(x='total_bill',y='tip',data = tips,kind = 'reg');
sns.jointplot(x='total_bill',y='tip',data = tips,kind = 'resid');
sns.jointplot(x='total_bill',y='tip',data = tips,kind = 'hex');
sns.pairplot(tips);
sns.pairplot(tips,hue='sex',palette='coolwarm');
sns.kdeplot(tips['total_bill'])
sns.rugplot(tips['total_bill']);
sns.kdeplot(tips['tip'])
sns.rugplot(tips['tip']);
import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()
tips.describe()
sns.barplot(x='sex',y='total_bill',data=tips);
import numpy as np
sns.barplot(x = 'sex',y = 'total_bill',data=tips,estimator=np.std);
sns.boxplot(x= 'day',y = 'total_bill',data=tips,palette = 'rainbow');
sns.boxplot(data = tips,palette = 'rainbow',orient = 'h'); #orient = h can do the plot for whole dataset
sns.boxplot(x = 'day',y = 'total_bill',hue = 'smoker',data = tips,palette = 'coolwarm');
sns.violinplot(x = 'day',y = 'total_bill',data=tips);
sns.violinplot(x = 'day',y = 'total_bill',hue = 'sex',data=tips,palette = 'Set1');
sns.violinplot(x = 'day',y = 'tip',hue='sex',data = tips,split = True,palette = 'Set1');
sns.stripplot(x = 'day',y = 'total_bill',data = tips);
sns.stripplot(x = 'day', y = 'total_bill' , hue = 'sex' , palette='Set1',data = tips);
sns.stripplot(x="day", y="total_bill", data=tips,jitter=True,hue='sex',palette='Set1',split=True);
sns.swarmplot(x = 'day' ,y = 'tip' , data  =tips);
sns.swarmplot(x="day", y="total_bill",hue='sex',data=tips, palette="Set1", split=True);
sns.violinplot(x="tip", y="day", data=tips,palette='rainbow')
sns.swarmplot(x="tip", y="day", data=tips,color='black',size=3);
flights = sns.load_dataset('flights')
tips = sns.load_dataset('tips')
flights.head()
tips.head()
tips.corr()
sns.heatmap(tips.corr());
sns.heatmap(tips.corr(),cmap='coolwarm',annot=True);
flights.pivot_table(values = 'passengers',index = 'month',columns = 'year')
pvflights = flights.pivot_table(values = 'passengers',index = 'month',columns = 'year')
sns.heatmap(pvflights);
sns.heatmap(pvflights,cmap='magma',linecolor='white',linewidths=1)
sns.clustermap(pvflights)
# More options to get the information a little clearer like normalization
sns.clustermap(pvflights,cmap='coolwarm',standard_scale=1)

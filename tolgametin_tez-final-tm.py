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
        
        
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib as mpl
from matplotlib import font_manager, rc
import seaborn as sns
%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
%matplotlib inline

data=pd.read_csv('../input/splitdata/mobility_cont.csv')
data['Series'].unique()
data['Series Code'].unique()
array=['Inbound mobility rate, both sexes (%)', 'Outbound mobility ratio, all regions, both sexes (%)', 'Total outbound internationally mobile tertiary students studying abroad, all countries, both sexes (number)', 'Net flow of internationally mobile students (inbound - outbound), both sexes (number)']
mobility=data.loc[data['Series Code'].isin(array)]
mobility['Country Name'].unique()
array3=['Arab World', 'East Asia & Pacific',
       'East Asia & Pacific (excluding high income)', 'Euro area',
       'Europe & Central Asia', 'Europe & Central Asia (excluding high income)', 'European Union',
       'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)',
       'Middle East & North Africa', 'Middle East & North Africa (excluding high income)',
       'North America', 'South Asia', 'Sub-Saharan Africa', 'Sub-Saharan Africa (excluding high income)']
mobility_continent=mobility.loc[mobility['Country Name'].isin(array3)]
mobility_continent
mobility_continent_in=mobility_continent[mobility_continent['Series Code'].str.contains('Inbound')]
x4=list(range(1997,2015))
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors=['black','r','darkred','gold','orange','brown','moccasin','m','darkmagenta','lightslategrey','darkgrey','b', 'indianred','lawngreen','darkgreen']
mobility_continent_out=mobility_continent[mobility_continent['Series Code'].str.contains('Outbound|outbound')].reset_index(drop=True)
mobility_continent_out['Country Name'].unique()
import matplotlib.style as style
style.use('default')
f = plt.figure(figsize=(16,24))
ax=f.add_subplot(311)
ax.set_title('Inbound mobility rate(%)', fontsize=16)
for i,j in zip(range(0,15),colors):
    ax.plot(x4,mobility_continent_in.iloc[i,3:],c=j)
ax.set_ylim([-0.3,8])
ax.set_xticks(np.arange(1997,2013,step=2))
ax.grid(linewidth=0.5)
ax.legend(mobility_continent_in['Code'])

ax2=f.add_subplot(312)
ax2.set_title('Outbound mobility rate(%)', fontsize=16)
for k,l in zip(range(0,15),colors):
    ax2.plot(x4, mobility_continent_out.iloc[2*k,3:],c=l)
ax2.set_ylim([0,7])
ax2.set_xticks(np.arange(1997,2013,step=2))
ax2.grid(linewidth=0.5)
ax2.legend(mobility_continent_out['Code'],loc='upper right', ncol=5)

ax3=f.add_subplot(313)
ax3.set_title('Total outbound internationally mobile students(number) (YEAR 2014)', fontsize=16)
for m,n in zip(range(0,15),colors):
    ax3.bar(mobility_continent_out.iloc[2*(m+1),1], mobility_continent_out.iloc[2*(m+1), 19], color=n)
ax3.set_ylim([0,1300000])
ax3.legend(mobility_continent_out['Code'],loc='upper right', ncol=3)
ax3.grid(linewidth=0.5)
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='India'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.1 India daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('India_daily_cases_19M52596.html')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Austria'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.2 Austlia daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Austlia_daily_cases_19M52596.html')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Brazil'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.3 Brazil daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Brazil_daily_cases_19M52596.html')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Japan'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.4 Japan daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Japan_daily_cases_19M52596.html')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='UK'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.5 UK daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('UK_daily_cases_19M52596.html')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Germany'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.6 Germany daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('Germany_daily_cases_19M52596.html')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='US'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
#print(np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)

df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()

from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily_recovery')


layout_object = go.Layout(title='Fig.7 US daily cases 19M52596',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)
iplot(fig)
fig.write_html('US_daily_cases_19M52596.html')

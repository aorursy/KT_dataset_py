import pandas as pd
import numpy  as np
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
tourist = pd.read_csv('../input/touristData.csv',encoding = "ISO-8859-1")
tourist.head()
pd.DataFrame(tourist.groupby(['Year'])['Count'].count()).iplot(kind='line', title='Evolution of Tourism in Brazil (Number of tourists throughout Time)',
                                                               yTitle='Number of Tourists', xTitle='Year', color='green')
pd.DataFrame(tourist.groupby(['Year', 'State'])['Count'].sum().sort_values().unstack())[21:].iplot(kind='line', title='Increase on Tourism by State',
                                                                                              xTitle='Year', yTitle='Number of Tourists')
pd.DataFrame(tourist.groupby(['WayIn','Year'])['Count'].sum().sort_values().unstack().transpose()).iplot(kind='scatter',filename='cufflinks/cf-simple-line',
                                                                                                        title='Tourists (Way in - Transportation) throughout time (1989-2015) in Brazil',
                                                                                                        xTitle='Year', yTitle='Number of Tourists')
pd.DataFrame(tourist.groupby(['WayIn','Year','Country'])['Count'].sum().sort_values().unstack().transpose())['Land'][list(range(1997,2003))].iplot(kind='bar')

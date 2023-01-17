import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import cufflinks as cf
from IPython.display import HTML
import seaborn as sns

import colorlover as cl
from IPython.display import HTML
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
%matplotlib inline
def cmocean_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

def cost_of_service_means(df_in):
    return df_in.groupby('cost_of_service')['actual_download','actual_upload','advertised_download','advertised_upload'].mean()
speedReport = pd.read_csv('../input/seattle-broadband-speed-test/broadband-speed-test.csv')
speed_df = pd.DataFrame(speedReport)
speed_df.head(10)
speed_df.max()
speed_df.min()
speed_df.info()
speed_df.describe()
speed_df.sort_values('actual_download', ascending=False)
speed_df.sort_values('actual_upload', ascending=False)
speed_df.sort_values('advertised_download', ascending=False)
speed_df.groupby('cost_of_service').count()
def __reindex__(df_in):
    df_in = df_in.reindex(index = ['100_or_above','75_100','50_75','25_50','less_than_25','dont_know'])
    return df_in

customerDistribution = speed_df.groupby('cost_of_service')['id'].count()
customerDistribution = __reindex__(customerDistribution)
customerDistribution.iplot(kind='bar', xTitle='Monthly cost of Internet Service', yTitle='Number of customers', title='Distribution of internet service subscribers in Seattle')
speed_df.groupby('isp').count().sort_values('id', ascending=False)
speed_df = speed_df[pd.notnull(speed_df['actual_download'])]
speed_df = speed_df[speed_df['actual_download'] != 0]
comcast_users = speed_df.query('isp == "comcast"')
centurylink_users = speed_df.query('isp == "centurylink"')
spectrum_users = speed_df.query('isp == "AS11404 vanoppen.biz LLC"')
wave_users = speed_df.query('isp == "wave"')
frontier_users = speed_df.query('isp == "frontier"')
top5Isp_providers = pd.concat([comcast_users,centurylink_users,spectrum_users,wave_users,frontier_users])
top5Isp_providers.sort_values(by=['advertised_upload'],ascending=False)
cost_of_service_means(top5Isp_providers)
top5Isp_providers.groupby('cost_of_service').min()
top5Isp_providers['timestamp'] = pd.to_datetime(top5Isp_providers.timestamp, unit='s')
advertisedOnly = top5Isp_providers[pd.notnull(top5Isp_providers['advertised_download'])]
top5Isp_providers = advertisedOnly[advertisedOnly.advertised_download > 0]
cost_of_service_means(top5Isp_providers)
top5Isp_providers[top5Isp_providers['cost_of_service'] == 'less_than_25'].sort_values('advertised_download', ascending=False)
top5Isp_providers.drop([3354], inplace=True)
speedsByCost = cost_of_service_means(top5Isp_providers)
speedsByCost = __reindex__(speedsByCost)
speedsByCost
speedsByCost.iplot(kind='bar',yTitle='Speed in mbps',xTitle='Price',title='Average measured speed vs Avg reported advertised speed (all ISPs)')
top5Isp_providers.iplot(kind='scatter', mode='markers',yTitle='Speed in mbps', x='cost_of_service', xTitle='Monthly Price of Service', y='actual_download', title='Actual Download Speeds vs Monthly Price of Service')
top5Isp_providers
pricingInfo = pd.read_csv('../input/isp-pricing/PricingInfo.csv')
pricingInfo = pricingInfo.rename(index=str, columns={'Unnamed: 0': 'Price Category'})
pricingInfo.head(5)
def __normalizePricing__(df_in, t1price, t2price, t3price, t4price, t5price):
    df_in.loc[df_in.cost_of_service == 'less_than_25', 'advertised_download'] = t1price
    df_in.loc[df_in.cost_of_service == '25_50', 'advertised_download'] = t2price
    df_in.loc[df_in.cost_of_service == '50_75', 'advertised_download'] = t3price
    df_in.loc[df_in.cost_of_service == '75_100', 'advertised_download'] = t4price
    df_in.loc[df_in.cost_of_service == '100_or_above', 'advertised_download'] = t5price
    return df_in
comcast_users = __normalizePricing__(comcast_users,0,37.5,200,700,1500)
centurylink_users = __normalizePricing__(centurylink_users,0,20,90,1000,1000)
spectrum_users = __normalizePricing__(spectrum_users,0,60,60,100,100)
top3Isp_providers = pd.concat([comcast_users,centurylink_users,spectrum_users])
top3Isp_providers['timestamp'] = pd.to_datetime(top3Isp_providers.timestamp, unit='s')
top3Isp_providers.info()
top3Isp_providersMeanCosts = cost_of_service_means(top3Isp_providers)
top3Isp_providersMeanCosts = __reindex__(top3Isp_providersMeanCosts)
top3Isp_providersMeanCosts = top3Isp_providersMeanCosts[top3Isp_providersMeanCosts.actual_download > 0]

top3Isp_providersMeanCosts.iplot(kind='bar',title='Avg. Measured Speed vs Max Speed Advertised (all ISPs)',yTitle='Speed in mbps',xTitle='Price',barmode='group')
comcast_advertisedVsActual = cost_of_service_means(comcast_users)
comcast_advertisedVsActual = __reindex__(comcast_advertisedVsActual)

centurylink_advertisedVsActual = cost_of_service_means(centurylink_users)
centurylink_advertisedVsActual = __reindex__(centurylink_advertisedVsActual)

spectrum_advertisedVsActual = cost_of_service_means(spectrum_users)
spectrum_advertisedVsActual = __reindex__(spectrum_advertisedVsActual)
comcast_users['actual_download'].max()
comcast_advertisedVsActual.iplot(kind='bar',colorscale='YlGnBu',title='Average measured speed vs Max advertised speed (Comcast)',yTitle='speed in mbps',xTitle='Price',barmode='group')
centurylink_users['actual_download'].max()
centurylink_advertisedVsActual.iplot(kind='bar',colorscale='set2',title='Average measured speed vs Max advertised speed (CenturyLink)',yTitle='Speed in mbps',xTitle='Price',barmode='group')
spectrum_users['actual_download'].max()
spectrum_users.sort_values(by=['advertised_upload'],ascending=False)
spectrum_advertisedVsActual.iplot(kind='bar', colorscale='accent',title='Average measured speed vs Max Advertised speed (Spectrum)',yTitle='Speed in mbps',xTitle='Price',barmode='group')

speeds_by_connection = speed_df.groupby(' connection_type')['actual_download','actual_upload'].mean()
speeds_by_connection.iplot(kind='bar',title='Average Measured speeds by Connection type',yTitle='Speed in mbps',xTitle='Connection Type')
top3Isp_providers.iplot(kind='scatter', mode='markers',yTitle='Speed in mbps', x=' connection_type', xTitle='Connection Type', y='actual_download', title='Actual Download Speeds vs Connection Type')
#top3Isp_providersMeanCosts = top3Isp_providers.groupby('cost_of_service')['actual_download','advertised_download'].mean()
wired_Top3Isp_providers = top3Isp_providers[top3Isp_providers[' connection_type'] == 'wired']
wired_Top3Isp_providers = cost_of_service_means(wired_Top3Isp_providers)
wired_Top3Isp_providers = __reindex__(wired_Top3Isp_providers)

wired_Top3Isp_providers.iplot(kind='bar',colorscale='YlGn',title='Average actual speed vs Advertised speed of Top 3 ISPs (Wired Connections)',yTitle='Speed in mbps',xTitle='Price',barmode='group')
peakHours = pd.DataFrame(top3Isp_providers)
offpeakHours = pd.DataFrame(top3Isp_providers)
top3Isp_byHour = pd.DataFrame(top3Isp_providers)
(top3Isp_providers['timestamp'].dt.hour).apply(pd.Series).iplot(kind='hist',
                                                                title='Number of Tests by Hour of Day',
                                                                xTitle='Hour of Day',
                                                                yTitle='Number of Tests')
peakHours.set_index('timestamp', inplace=True)
peakHours = peakHours.between_time('19:00','23:00')
offpeakHours = offpeakHours.between_time('23:00','19:00') 

peakHours_Means = __reindex__(peakHours.groupby('cost_of_service')['actual_download','actual_upload'].mean())
peakHours_Means.iplot(kind='bar',title='Average measured speed vs Max advertised speed during Peak Hours (Top 3 ISPs)',yTitle='Speed in mbps',xTitle='Price',barmode='group')

offpeakHours_Means = __reindex__(offpeakHours.groupby('cost_of_service')['actual_download','actual_upload'].mean())
offpeakHours_Means.iplot(kind='bar',title='Average measured speed vs Max advertised speed during Off Peak Hours (Top 3 ISPs)',yTitle='Speed in mbps',xTitle='Price',barmode='group')
top3Isp_byHour
top3Isp_providers.index=top3Isp_providers.index.strftime('%H')
top3Isp_speedsByHour = top3Isp_providers['actual_download'].apply(pd.Series)
df = top3Isp_speedsByHour[0].apply(pd.Series)
df.iplot(kind='scatter',mode='markers',xTitle='Hour of Day',yTitle='Speeds in Mbps',title='Actual Speeds Recorded by Hour of Day')


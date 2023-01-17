from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests

IP_COLUMN_NAME = 'ip'
IP_DETAILS_EP = "https://free.ipdetails.io"

def enrich_server_details(ipaddress):
    hosting_country, hosting_provider,hosting_cdn  = None, None, None 
    base_url = IP_DETAILS_EP
    endpoint = f"{base_url}/{ipaddress}"
    # ipdetails.io provides unlimited free requests - good to have one like this    
    r = requests.get(endpoint)    
    if r.status_code not in range(200, 299):
        return None, None, None
    try:
        '''
        This try block incase any of our inputs are invalid. This is done instead
        of actually writing out handlers for all kinds of responses.
        '''
        results = r.json()        
        hosting_country = (results['country'])['country_short']
        hosting_provider = ""
        hosting_cdn = ""
    except:
        pass
    return hosting_country, hosting_provider, hosting_cdn
    
def enrich_data(row):
    ipaddress = row[IP_COLUMN_NAME]
    srvr_country, srvr_cloud, srvr_cdn = enrich_server_details(ipaddress)
    row['hosting_country'] = srvr_country
    row['hosting_provider'] = srvr_cloud
    row['hosting_cdn'] = srvr_cdn   
    return row

def make_master_ds():    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            df1 = pd.read_csv(os.path.join(dirname, filename), delimiter=',', nrows = 1)
            mDS.append(df1)
    
mDS = [];
make_master_ds()
wrkngFrame = pd.concat(mDS, axis=0, ignore_index=True)
wrkngFrame.dataframeName = 'world.ecommerce.websites'

# accomodate new columns
wrkngFrame['hosting_country'], wrkngFrame['hosting_provider'], wrkngFrame['hosting_cdn'] =  None, None, None

# enrich hosting details
wrkngFrame.apply(enrich_data, axis=1) 

wrkngFrame

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from tld import get_tld

DOMAIN_COLUMN_NAME = 'domain'
COUNTRY_OF_INTEREST = 'pk'

def extract_tld(domainname):
    return get_tld(domainname, fail_silently=True, fix_protocol=True)
    
def enrich_data(row):
    domainame = row[DOMAIN_COLUMN_NAME]   
    row['domaincountry'] = extract_tld(domainame)
    return row

def make_master_ds():    
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            # remember to REMOVE the nrows argument, to get the entire list
            df1 = pd.read_csv(os.path.join(dirname, filename), delimiter=',', nrows = 5000)
            mDS.append(df1)
    
mDS = [];
make_master_ds()
wrkngFrame = pd.concat(mDS, axis=0, ignore_index=True)
wrkngFrame.dataframeName = 'world.ecommerce.websites'

# accomodate new columns
wrkngFrame['domaincountry'] =  None

# enrich hosting details
wrkngFrame.apply(enrich_data, axis=1) 

countryofinterest = wrkngFrame[wrkngFrame['domaincountry'] == COUNTRY_OF_INTEREST]                               
countryofinterest
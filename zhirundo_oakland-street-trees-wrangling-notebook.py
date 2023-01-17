#imports libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import datetime as dt
#displays 500 rows before auto collapsing - used for visual analysis
pd.set_option('display.max_rows', 5000)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv(('../input/oakland-street-trees.csv'), index_col=False)
#changes column names to lower case
df.columns = df.columns.str.lower()
#extracts lat/long into separate columns and drops unnecessary columns
df['latitude_1'] = df['location 1'].str.extract('(\'\d\d\.\d\d\d\d\d\d\d\d\d\d\d\d\d\d)', expand=True)
df['latitude'] = df['latitude_1'].str.extract('(\d\d\.\d\d\d\d\d\d\d\d\d\d\d\d\d\d)', expand=True)
df['longitude'] = df['location 1'].str.extract('(-\d\d\d\.\d\d\d\d\d\d\d\d\d\d\d\d\d\d)', expand=True)
df.drop(columns={'latitude_1', 'location 1', 'stname'}, inplace=True)
#changes names to lower case and extracts genus from species column
df['species'] = df['species'].str.lower()
df['species'] = df['species'].str.replace(' sp', '')
df['genus_name'] = df['species'].str.extract('([a-z]\w{0,})')
#changes common for latin names
df['genus_name'] = df['genus_name'].str.replace('walnut', 'juglans')
df['genus_name'] = df['genus_name'].str.replace('fig', 'ficus')
df['genus_name'] = df['genus_name'].str.replace('banana', 'musa')
df['genus_name'] = df['genus_name'].str.replace('apricot', 'prunus')
df['genus_name'] = df['genus_name'].str.replace('almond', 'prunus')
df['genus_name'] = df['genus_name'].str.replace('tbd', 'NaN')
df['genus_name'] = df['genus_name'].str.replace('other', 'NaN')
df['genus_name'] = df['genus_name'].str.replace('unknown', 'NaN')
#drops species column
df.to_csv('oakland_street_trees.csv')

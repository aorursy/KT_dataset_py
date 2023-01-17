
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
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

print('Modules are imported.')
corona_dataset_csv = pd.read_csv('/kaggle/input/immuni/covid19.csv')
corona_dataset_csv.head(10)
corona_dataset_csv.info()

corona_dataset_csv.describe()
corona_dataset_csv.corr(method='pearson')
corona_dataset_csv.corr(method ='kendall') 
corona_dataset_csv.corr(method ='spearman') 
x = corona_dataset_csv['Immune DP1']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune DP3']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune HEPB3']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune HEPBB']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune HIB3']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune MCV1']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune MCV2']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune PCV3']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune POL3']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune ROTAC']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune RCV1']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune YFV']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune BCG']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune IPV']
y = corona_dataset_csv['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune DP1']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune DP3']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune HEPB3']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune HEPBB']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune HIB3']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune MCV1']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune MCV2']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune PCV3']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune POL3']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune ROTAC']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune RCV1']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})

x = corona_dataset_csv['Immune YFV']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune BCG']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
x = corona_dataset_csv['Immune IPV']
y = corona_dataset_csv['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "black"}, line_kws={"color": "red"})
df=corona_dataset_csv[['Immune PCV3','Immune YFV','Immune ROTAC']]
df.fillna(1)
df
df['Combo1']=df['Immune PCV3']+df['Immune YFV']+df['Immune ROTAC']

df['Mortality']=corona_dataset_csv['Mortality']
df['Morbidity']=corona_dataset_csv['Morbidity']
df.head()

df.corr(method='pearson')
df['Combo2']=df['Immune PCV3']+df['Immune YFV']
df.corr(method='pearson')
df['Combo3']=df['Immune YFV']+df['Immune ROTAC']
df.corr(method='pearson')
df['Combo4']=df['Immune PCV3']+df['Immune ROTAC']
df.corr(method='pearson')
x = df['Combo1']
y = df['Morbidity']
sns.regplot(x,np.log(y),scatter_kws={"color": "blue"}, line_kws={"color": "black"})
x = df['Combo1']
y = df['Mortality']
sns.regplot(x,np.log(y),scatter_kws={"color": "blue"}, line_kws={"color": "black"})
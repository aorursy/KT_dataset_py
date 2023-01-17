#import the standard data analytics libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline

sns.set() #set style for any plots



#import the data

data = pd.read_csv('/kaggle/input/gas-prices-in-brazil/2004-2019.tsv',sep='\t')

data.columns # output the colums
#drop irrelevant column

data.drop('Unnamed: 0',axis=1,inplace=True)
# For easy interpretability, we translate the columns to english



data.rename(

columns={

        "DATA INICIAL": "start_date",

        "DATA FINAL": "end_date",

        "REGIÃO": "region",

        "ESTADO": "state",

        "PRODUTO": "product",

        "NÚMERO DE POSTOS PESQUISADOS": "no_gas_stations",

        "UNIDADE DE MEDIDA": "unit",

        "PREÇO MÉDIO REVENDA": "avg_price",

        "DESVIO PADRÃO REVENDA": "sd_price",

        "PREÇO MÍNIMO REVENDA": "min_price",

        "PREÇO MÁXIMO REVENDA": "max_price",

        "MARGEM MÉDIA REVENDA": "avg_price_margin",

        "ANO": "year",

        "MÊS": "month",

        "COEF DE VARIAÇÃO DISTRIBUIÇÃO": "coef_dist",

        "PREÇO MÁXIMO DISTRIBUIÇÃO": "dist_max_price",

        "PREÇO MÍNIMO DISTRIBUIÇÃO": "dist_min_price",

        "DESVIO PADRÃO DISTRIBUIÇÃO": "dist_sd_price",

        "PREÇO MÉDIO DISTRIBUIÇÃO": "dist_avg_price",

        "COEF DE VARIAÇÃO REVENDA": "coef_price"

    },

    inplace=True

)
# view data

data.head()
# convert columns that should be numeric to numbers

names = ['avg_price_margin','coef_price','dist_avg_price','dist_sd_price','dist_min_price','dist_max_price','coef_dist']



for col in names:

    data[col]=pd.to_numeric(data[col],errors='coerce')



data.dtypes
#determine if the are any missing values

# find the shape of the data

shape = data.shape

missing = data.isnull().any()



print(shape,'\n')

print(missing)
# Check how many null values there are in each of the columns that came up missing values in the previous cell.

margins_missing = data['avg_price_margin'].isnull().sum()

avg_price_missing = data['dist_avg_price'].isnull().sum()

sd_price_missing = data['dist_sd_price'].isnull().sum()

dist_min_price_missing = data['dist_min_price'].isnull().sum()

dist_max_price_missing = data['dist_max_price'].isnull().sum()

coef_dist_missing = data['coef_dist'].isnull().sum()



# Print the number of missing values in each column

print(margins_missing)

print(avg_price_missing)

print(sd_price_missing)

print(dist_min_price_missing)

print(dist_max_price_missing)

print(coef_dist_missing)



# Drop every entry with missing values

data.dropna(axis=0,inplace=True)
# Determine the number and names of the products sold in Brazil

products = data['product'].unique()

print(len(products))

print(products)
# Find the number and names of the states in Brazil

gnv_data = data[data['product'] == 'GNV']

states = gnv_data['state'].unique()

states
# Plot the distribution of the average price in each state

# Although anova is robust to non normally distributed data, It is good to know if the assumption of normality holds.

fig,ax = plt.subplots(6,4,figsize=(10,10),constrained_layout=True)

ax = ax.ravel()



for i in range(len(states)):

    ax[i].hist(gnv_data[gnv_data.state == states[i]]['avg_price'])

    ax[i].set_title(states[i])

    ax[i].set_xlabel('avg price')

# Determine how many observations we have from every state

gnv_state_count = gnv_data.groupby('state')['avg_price'].count()

gnv_state_count
# separate the data into two sets, gnv_data2 has the states with less than 100 obseravtions removed 

low_count_states = ['AMAPA','DISTRITO FEDERAL','GOIAS','MARANHAO','PARA','PIAUI','TOCANTINS']

low_count_states_df = gnv_data[gnv_data['state'].isin(low_count_states)]

gnv_data2 = gnv_data.drop(low_count_states_df.index,axis=0)

states2 = np.setdiff1d(states,low_count_states) # Create a list of the remaining states 

print(states2)

print(len(states2))
# We determine if the data is normally distributed, by way of qqplot 

# beacuse the histogram did not provide any useful information

from statsmodels.graphics.gofplots import qqplot



fig,ax = plt.subplots(6,3,figsize=(10,10),constrained_layout=True)

ax = ax.ravel()



for i in range(len(states2)):

    qqplot(gnv_data[gnv_data.state == states2[i]]['avg_price'],line='s',ax=ax[i])

    ax[i].set_title(states2[i])

    ax[i].set_xlabel('avg price')



# Run a shapiro normality test at a 5% significance level

from scipy.stats import shapiro



alpha = 0.05

reject_count = 0 # count of all the states that dont have normally distributed data

normal_count = 0 # count of states with normally distributed data



for i in range(len(states2)):

    

    stat, p = shapiro(gnv_data2[gnv_data.state == states2[i]]['avg_price'])

    #print(states2[i])

    #print('Statistics=%.3f, p=%.3f' % (stat, p))



    if p > alpha:

        normal_count += 1

    else:

        reject_count += 1



print('number of rejects =',reject_count)

print('number of normally distributed prices =',normal_count)
# Next we run a normaltest to verify the results of the previous test.

from scipy.stats import normaltest



alpha = 0.05

reject_count = 0

normal_count = 0



for i in range(len(states2)):

    

    stat, p = normaltest(gnv_data2[gnv_data.state == states2[i]]['avg_price'])

    #print(states2[i])

    #print('Statistics=%.3f, p=%.3f' % (stat, p))



    if p > alpha:

        normal_count += 1

    else:

        reject_count += 1



print('number of rejects =',reject_count)

print('number of normally distributed prices =',normal_count)
# Next we run anova using the states as treatments and the avg_price as the response variable

import statsmodels.api as sm

from statsmodels.formula.api import ols



samples = pd.DataFrame(columns=gnv_data2.columns) # Create a DataFrame to store the samples



for state in states2:

    sample = gnv_data2[gnv_data2.state == state].sample(100) # Sample 100 values from each state.

    samples = pd.concat([samples,sample])





model = ols('avg_price ~ state', data=samples).fit()

anova_table = sm.stats.anova_lm(model,typ=3)



print(model.summary())

print()

print(anova_table)
# Next we attempt to discern more about the r_squared value for the anova test. 

# 5,000 repitions of 100 bootstrapped samples (10,000 waaay too slow) will be used to find a distribution for the value.

# The calculations will be hard coded for this part of the analysis.



N = 5000

s = 100

P = len(states2)

n = len(states2) - 1

R_squared = []

R_squared_a = []

for i in range(N):

    samples = pd.DataFrame(columns=gnv_data2.columns)

    for state in states2:

        sample = gnv_data2[gnv_data2.state == state].sample(s,replace=True)

        samples = pd.concat([samples,sample])

    

    state_means = samples.groupby('state')['avg_price'].mean()

    overall_mean = samples['avg_price'].mean()

    

    SSA = (s*((state_means - overall_mean)**2)).sum() # Sum squared treatments

    MSA = SSA/n # Mean square treatments

    SST = ((samples['avg_price']-overall_mean)**2).sum() # Total Sum squares 

    MST = SST/(P*s - 1)

    SSE = SST - SSA # Sum squared residuals

    MSE = SSE/(P*(s-1))

    r_2 = SSA/SST # R_squared

    r_2a = 1-MSE/MST # Adjusted R_squared

    R_squared.append(r_2)

    R_squared_a.append(r_2a)

    

mean_r2 = np.mean(R_squared)

mean_r2a = np.mean(R_squared_a)



fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].hist(R_squared)

ax[0].set_title('Sampling distribution of R_squared')

ax[0].set_xlabel('R_square value')

ax[0].set_ylabel('frequency')



ax[1].hist(R_squared_a)

ax[1].set_title('Sampling distribution of adjusted R_squared')

ax[1].set_xlabel('R_square value')

ax[1].set_ylabel('frequency')



print('R_squared mean value is %.3f and the adjusted R-squared mean value is %.3f' % (mean_r2,mean_r2a))
# Pair wise comparison of the top five states by avg_price



pair_comp = model.t_test_pairwise('state')

pair_comp_df = pair_comp.result_frame



#pair_comp_df



results = pd.DataFrame([pair_comp_df.loc['RIO GRANDE DO SUL-AMAZONAS'],pair_comp_df.loc['PARAIBA-AMAZONAS'],

                       pair_comp_df.loc['SERGIPE-AMAZONAS'],pair_comp_df.loc['MATO GROSSO DO SUL-AMAZONAS'],

                       pair_comp_df.loc['RIO GRANDE DO SUL-MATO GROSSO DO SUL'],pair_comp_df.loc['PARAIBA-MATO GROSSO DO SUL'],

                       pair_comp_df.loc['SERGIPE-MATO GROSSO DO SUL'],pair_comp_df.loc['SERGIPE-PARAIBA'],

                       pair_comp_df.loc['RIO GRANDE DO SUL-PARAIBA'],pair_comp_df.loc['SERGIPE-RIO GRANDE DO SUL'] ])

results
for region in gnv_data2['region'].unique():

    sample = gnv_data2[gnv_data2.region == region].sample(100) # Sample 100 values from each state.

    samples = pd.concat([samples,sample])





model = ols('avg_price ~ region', data=samples).fit()

anova_table = sm.stats.anova_lm(model,typ=3)



print(model.summary())

print()

print(anova_table)
# Next we attempt to discern more about the r_squared value for the anova test. 

# 5,000 repitions of 100 bootstrapped samples (10,000 waaay too slow) will be used to find a distribution for the value.

# The calculations will be hard coded for this part of the analysis.



N = 5000

s = 100

P = len(states2)

n = len(states2) - 1

R_squared = []

R_squared_a = []

for i in range(N):

    samples = pd.DataFrame(columns=gnv_data2.columns)

    for region in gnv_data2['region'].unique():

        sample = gnv_data2[gnv_data2.region == region].sample(s,replace=True)

        samples = pd.concat([samples,sample])

    

    region_means = samples.groupby('region')['avg_price'].mean()

    overall_mean = samples['avg_price'].mean()

    

    SSA = (s*((region_means - overall_mean)**2)).sum() # Sum squared treatments

    MSA = SSA/n # Mean square treatments

    SST = ((samples['avg_price']-overall_mean)**2).sum() # Total Sum squares 

    MST = SST/(P*s - 1)

    SSE = SST - SSA # Sum squared residuals

    MSE = SSE/(P*(s-1))

    r_2 = SSA/SST # R_squared

    r_2a = 1-MSE/MST # Adjusted R_squared

    R_squared.append(r_2)

    R_squared_a.append(r_2a)

    

mean_r2 = np.mean(R_squared)

mean_r2a = np.mean(R_squared_a)



fig,ax = plt.subplots(1,2,figsize=(15,5))

ax[0].hist(R_squared)

ax[0].set_title('Sampling distribution of R_squared')

ax[0].set_xlabel('R_square value')

ax[0].set_ylabel('frequency')



ax[1].hist(R_squared_a)

ax[1].set_title('Sampling distribution of adjusted R_squared')

ax[1].set_xlabel('R_square value')

ax[1].set_ylabel('frequency')



print('R_squared mean value is %.3f and the adjusted R-squared mean value is %.3f' % (mean_r2,mean_r2a))
def capitalize_after_hyphen(x):
    a=list(x)
    a[p.index('-')+1]=a[p.index('-')+1].capitalize()
    x=''.join(a)
    return ''.join(a)

import pandas as pd
import requests  
#l=['patients','admdissions','diagnoses','drg-codes','icu-stays','procedures','prescriptions','d-icd-diagnoses','d-icd-procedures']
url1="http://ec2-54-88-151-77.compute-1.amazonaws.com:3004/v1/hrrd-table?limit=10000&offset=0"

d={}
url=[url1]

for x in url:  
    p = x[(x.index('v1/')+len('v1/')):x.index('?limit')]
    if p=='state-codes':
        p='stateCode'
    else:
        
        try:
            p=capitalize_after_hyphen(p)
        except:
            pass
        try:
            p=p[:p.index('-')]+p[p.index('-')+1:]
        except:
            pass

        try:
            p=capitalize_after_hyphen(p)
        except:
            pass
        try:
            p=p[:p.index('-')]+p[p.index('-')+1:]
        except:
            pass
    
    
    
    print(p)
    
    d['{}'.format(p)]=pd.DataFrame(requests.get(x).json()['{}'.format(p)])

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bkp
import scipy.stats as stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
hospital_read_df = d['hrrdTable']
hospital_read_df.dtypes
# deal with missing and inconvenient portions of data 
clean_hospital_read_df = hospital_read_df[hospital_read_df['number_of_discharges'] != 'Not Available']
clean_hospital_read_df.loc[:, 'number_of_discharges'] = clean_hospital_read_df['number_of_discharges'].astype(int)
clean_hospital_read_df = clean_hospital_read_df.sort_values('number_of_discharges')
print(len(clean_hospital_read_df))
clean_hospital_read_df.head()
clean_hospital_read_df=clean_hospital_read_df.replace('Not Available',np.nan)
for x in clean_hospital_read_df.columns:
    try:
        clean_hospital_read_df[x]=clean_hospital_read_df[x].astype(float)
    except:
        pass
# generate a scatterplot for number of discharges vs. excess rate of readmissions
# lists work better with matplotlib scatterplot function - Series data can work, too
x = clean_hospital_read_df['number_of_discharges'][81:-3]
y = clean_hospital_read_df['excess_readmission_ratio'][81:-3]

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x, y,alpha=0.2)

ax.fill_between([0,350], 1.15, 2, facecolor='red', alpha = .15, interpolate=True)
ax.fill_between([800,2500], .5, .95, facecolor='green', alpha = .15, interpolate=True)

ax.set_xlim([0, max(x)])
ax.set_xlabel('number_of_discharges', fontsize=12)
ax.set_ylabel('excess_readmission_ratio', fontsize=12)
ax.set_title('Scatterplot of number of discharges vs. excess rate of readmissions', fontsize=14)

ax.grid(True)
fig.tight_layout()
#null hypothesis : there is not a significant difference in hospitals/facilities with discharges > 1000 and discharges < 100
#alternative :  there is a difference, and that the hospitals with discharges < 100 have a higher readmission rate
low_discharge_df  = clean_hospital_read_df['excess_readmission_ratio'][81:-3].loc[
    clean_hospital_read_df['number_of_discharges'] <= 100]
high_discharge_df = clean_hospital_read_df['excess_readmission_ratio'][81:-3].loc[
    clean_hospital_read_df['number_of_discharges'] >= 1000]
print(low_discharge_df.head())
print(high_discharge_df.head())
print(len(low_discharge_df))
print(len(high_discharge_df))
low_discharge_df = low_discharge_df.astype(float)
high_discharge_df = high_discharge_df.astype(float)
print('Low Discharge Rate mean and std Readmission Ratio:',np.mean(low_discharge_df), np.std(low_discharge_df))
print('High Discharge Rate mean and std Readmission Ratio:', np.mean(high_discharge_df), np.std(high_discharge_df))
# t-test
t, p = stats.ttest_ind(low_discharge_df, high_discharge_df, equal_var=False)
print("ttest_ind:            t = %g  p = %g" % (t, p))
# Effect Size
def CohenEffectSize(group1, group2):
    """Compute Cohen's d.

    group1: Series or NumPy array
    group2: Series or NumPy array

    returns: float
    """
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d
CohenEffectSize(low_discharge_df, high_discharge_df)
# Code for calculating overlap and superiority between two groups
import scipy.stats
def overlap_superiority(control, treatment, n=1000):
    """Estimates overlap and superiority based on a sample.
    
    control: scipy.stats rv object
    treatment: scipy.stats rv object
    n: sample size
    """
    control_sample = control
    treatment_sample = treatment
    thresh = (control.mean() + treatment.mean()) / 2
    
    control_above = sum(control_sample > thresh)
    treatment_below = sum(treatment_sample < thresh)
    overlap = (control_above + treatment_below) / n
    
    superiority = sum(x > y for x, y in zip(treatment_sample, control_sample)) / n
    return overlap, superiority
overlap_superiority(low_discharge_df, high_discharge_df, n=461)
plt.hist(low_discharge_df, bins=50, alpha=0.5, label='Low Discharge')
plt.hist(high_discharge_df, bins=50, alpha=0.5, label='High Discharge')
plt.legend(loc='upper right')
plt.show()

clean_hospital_read_df.drop(['footnote'],axis=1,inplace=True)
import seaborn as sns
clean_hospital_read_df.corr(method='pearson', min_periods=1)
clean_hospital_read_df.head(2)
clean_hospital_read_df['expected_readmission_rate'].dtype
float_dataframe=clean_hospital_read_df.loc[:,clean_hospital_read_df.dtypes==float]
clean_hospital_read_df.corr(method='spearman')
for x in float_dataframe.columns:
    for y in float_dataframe.columns:
        sns.lmplot(x=x,y=y,data=clean_hospital_read_df,fit_reg=True)
sns.heatmap(clean_hospital_read_df.corr())
##There is a significant correlation between hospital capacity (number of discharges) and readmission rates.
##Smaller hospitals/facilities may be lacking necessary resources to ensure quality care and prevent complications that lead to readmissions.
# Import all required modules
import pandas as pd
import numpy as np

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import plotting modules
import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
%matplotlib inline
# Tune the visual settings for figures in `seaborn`
sns.set_context(
    "notebook", 
    font_scale=1.5,       
    rc={ 
        "figure.figsize": (11, 8), 
        "axes.titlesize": 18 
    }
)

from matplotlib import rcParams
rcParams['figure.figsize'] = 11, 8
df = pd.read_csv('../input/mlbootcamp5_train.csv')
print('Dataset size: ', df.shape)
df.head()
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active', 'cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=12);
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active'], 
                     id_vars=['cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 
                                              'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               col='cardio', data=df_uniques, kind='bar', size=9);
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
#sns.barplot(x='height', hue='gender', data=df)
df.groupby('gender')['height'].mean()
df.gender.value_counts()
sns.countplot(x='alco', hue='gender', data=df)
df.groupby('gender')['smoke'].mean()
df.groupby('smoke')['age'].median() / 365
df[df.smoke == 0]['age'].median() / 30 - df[df.smoke == 1]['age'].median() / 30 
# You code here
df['age_years'] = round(df['age'] / 365)
df_to_test = df[(df.age_years >= 60) & (df.age_years <= 64) & (df.gender == 2) & (df.smoke == 1)]
df_to_test.head()
print(df_to_test[(df_to_test.ap_hi < 120) & (df_to_test.cholesterol == 1)]['cardio'].mean())
print(df_to_test[(df_to_test.ap_hi >= 160) & (df_to_test.ap_hi < 180) & (df_to_test.cholesterol == 3)]['cardio'].mean())
df_to_test[(df_to_test.ap_hi < 120) & (df_to_test.cholesterol == 1)].count()
df_to_test[(df_to_test.ap_hi >= 160) & (df_to_test.ap_hi < 180) & (df_to_test.cholesterol == 3)]
# You code here
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
df['BMI'].median()
df.groupby('gender').BMI.mean()
df.groupby('cardio').BMI.mean()
df.groupby(['gender', 'alco', 'cardio']).BMI.median()
# You code here
df_clean = df[(df.ap_lo <= df.ap_hi) & (df.height >= df.height.quantile(0.025)) & (df.height <= df.height.quantile(0.975))
             & (df.weight >= df.weight.quantile(0.025)) & (df.weight <= df.weight.quantile(0.975))]
df.shape[0], df_clean.shape[0], df_clean.shape[0] / df.shape[0]
# You code here
sns.heatmap(df.corr(method='pearson'), annot=True, fmt=".1f", linewidths=.5)
# You code here
df_ = pd.melt(df, value_vars=['height', 'weight'], id_vars='gender')
sns.violinplot(x='variable', y='value', hue='gender', data=df_,  split=True, scale='count', scale_hue=False)
plt.show()
# You code here
sns.heatmap(df.corr(method='spearman'), annot=True, fmt=".1f", linewidths=.5)
# You code here
plt.figure(figsize=(20,12))
sns.countplot(x="age_years", hue='cardio', data=df)
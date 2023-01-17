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
df_uniques

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
df.head()
gender1 = df[df['gender'] == 1]['height'].mean()
gender2 = df[df['gender'] == 2]['height'].mean()
print('The 1st gender height is {:.2f} sm \nThe 2nd gender height is {:.2f} sm.'.format(gender1, gender2))

# So, probably, gender 2 is a male gender
# Let's count gender values and get the answer.
df['gender'].value_counts()

# Answer: 1
sns.countplot(x='gender', hue='alco', data=df);
df.groupby(['gender'])['alco'].value_counts(normalize=True)

# As we can see from the table and the plot, men (gender 2) reports consuming alcohol more often.
# Answer: 2
sns.countplot(x='gender', hue='smoke', data=df);

smoking_women = ( (df['gender'] == 1) & (df['smoke'] == 1) ).sum() / (df['gender'] == 1).sum()
smoking_men = ( (df['gender'] == 2) & (df['smoke'] == 1) ).sum() / (df['gender'] == 2).sum()
share_diff = smoking_men - smoking_women

print('Difference between the percentages of smokers among men and women equals: {:.0%}'.format(share_diff))
# Answer: 3
nonsmokers_age = df[df['smoke'] == 0]['age'].median() / 30
smokers_age = df[df['smoke'] == 1]['age'].median() / 30
age_diff = round(nonsmokers_age - smokers_age)
print('The difference between median values of age for smokers and non-smokers is {} months'.format(age_diff))

# Answer: 4
new_data = df.copy()
age_in_years = round(df['age'] / 365.25).astype(int)
new_data.insert(2, 'age_years', age_in_years)


ap_hi_cat = pd.cut(new_data['ap_hi'], 
                   bins=[0, 120, 160, 180], 
                   labels=[1, 2, 3],
                  include_lowest=True,
                  right=False)

new_data.insert(7, 'ap_hi_cat', ap_hi_cat)

# Let's create a temporary subset to answer the following questions
temp_df = new_data[(new_data['age_years'] >= 60) & 
                   (new_data['age_years'] <= 64) &
                   (new_data['gender'] == 2) &
                   (new_data['smoke'] == 1)]

temp_df.head()

# Or we can use less elegant solution:
# def ap_hi_to_categorical(ap_hi):
#     if ap_hi < 120:
#         return 'low'
#     elif 160 <= ap_hi < 180:
#         return 'high'
#     return 'normal'
# age_years['ap_hi_cat'] = age_years['ap_hi'].apply(ap_hi_to_categorical)
# We can see the fractions using groupby()
temp_df.groupby(['cholesterol', 'ap_hi_cat'])['cardio'].value_counts(normalize=True).to_frame()

# Or count them directly:
low_ap_and_cholesterol_with_cardio_share = temp_df[(temp_df['cholesterol'] == 1) & 
          (temp_df['ap_hi_cat'] == 1)]['cardio'].value_counts(normalize=True)[1]
high_ap_and_cholesterol_with_cardio_share = temp_df[(temp_df['cholesterol'] == 3) & 
          (temp_df['ap_hi_cat'] == 3)]['cardio'].value_counts(normalize=True)[1]

fractions_ratio = high_ap_and_cholesterol_with_cardio_share / low_ap_and_cholesterol_with_cardio_share
print('Fraction ratio is {:.0f}'.format(fractions_ratio))

# Answer: 3
bmi = new_data['weight'] / (new_data['height'] / 100)**2
new_data.insert(5, 'BMI', bmi)
new_data.head()
# Statement 1:
median_bmi = new_data['BMI'].median() 
18.5 <= median_bmi <= 25 # False
# Statement 2:
new_data[new_data['gender'] == 1]['BMI'].mean() > new_data[new_data['gender'] == 2]['BMI'].mean() # True 
# Statement 3:
new_data[new_data['cardio'] == 0]['BMI'].mean() > new_data[new_data['cardio'] == 1]['BMI'].mean() # False

# Also, we can use a plot:
plt.ylim(10, 60) # We limit Y-axis values so outliers do not distort our graph too much
sns.boxplot(x='cardio', y='BMI', data=new_data);

# We can see that people with the higher BMI tend to have little more CVD. Again, the statement is False.
# Statement 4

new_data.groupby(['gender', 'alco', 'cardio'])['BMI'].mean().to_frame()

nondrinking_men = new_data[(new_data['gender'] == 2) & 
                           (new_data['alco'] == 0) &
                           (new_data['cardio'] == 0)]['BMI'].mean()
nondrinking_women = new_data[(new_data['gender'] == 1) & 
                             (new_data['alco'] == 0) &
                             (new_data['cardio'] == 0)]['BMI'].mean()
print('Mean BMI for non-drinking men:', round(nondrinking_men, 1))
print('Mean BMI for non-drinking women:', round(nondrinking_women, 1))

# Thus, the mean BMI value for healthy men is closer to normal BMI compared to healthy women.
# Thus, the 4th statement is True

# Answers: 2, 4
# # Just for practice, there're 2 ways:
# # 1) We're dropping rows where diastolic pressure is higher than systolic:
# filt_df = new_data[new_data['ap_hi'] > new_data['ap_lo']]

# # 2) We can do the same through drop() as well:
# dirt_data = new_data[new_data['ap_hi'] <= new_data['ap_lo']]
# clean_data = new_data.drop(dirt_data.index)

# filt_df.equals(clean_data) # Check if subsets are equal. Returns True.
# We've already dropped rows with ap_hi > ap_lo. Next step:
filt_df = new_data[(new_data['ap_hi'] > new_data['ap_lo']) & 
                  (new_data['height'] >= new_data['height'].quantile(0.025)) &
                  (new_data['height'] <= new_data['height'].quantile(0.975)) & 
                  (new_data['weight'] >= new_data['weight'].quantile(0.025)) & 
                  (new_data['weight'] <= new_data['weight'].quantile(0.975))]
filt_df.shape[0] / df.shape[0]

# Answer: 3
corr = filt_df.corr()
sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.5, center=0);
# From the correlation matrix we can see that 'Height' and 'Smoke' features 
# has the strongest correlation with gender feature
# Answer: 2
df_melt = pd.melt(frame=filt_df, value_vars=['height'], id_vars=['gender'])
df_melt
sns.violinplot(x='variable',y='value', hue='gender', data=df_melt, split=True, scale='count', scale_hue=False);
# let's select required features for a better view:
features = ['height', 'weight','age', 'cholesterol', 'gluc', 'cardio', 'ap_hi', 'ap_lo', 'smoke', 'alco']
corr = filt_df[features].corr(method='spearman')
sns.heatmap(corr, annot=True, fmt='.1f', linewidths=.5, center=0);

# We can see that among given pairs, Ap_hi - Ap_lo pair has the strongest Spearman correlation
# Answer: 5
rank_corr = filt_df.corr(method='spearman') 
sns.heatmap(rank_corr, annot=True, fmt='.1f', linewidths=0.5, center=0);
plt.xlim(0, 200)
plt.ylim(0, 300)
plt.scatter(x=filt_df['ap_lo'], 
            y=filt_df['ap_hi']);
plt.xticks(rotation=90)
sns.countplot(x="age_years", hue='cardio', data=filt_df);
# Answer: 2
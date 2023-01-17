import numpy as np
import pandas as pd 

from mlxtend.frequent_patterns import apriori, association_rules
from IPython.display import Markdown, display

def write_markdown(filename, df):
    fmt = ['---' for i in range(len(df.columns))]
    df_fmt = pd.DataFrame([fmt], columns=df.columns)
    df_formatted = df
    
     # lose the frozensets
    df_formatted = pd.DataFrame(df)
    for column in df_formatted:
        if column in ['itemsets', 'antecedants', 'consequents']:
            df_formatted[column] = df_formatted[column].apply(lambda x: list(x))

    df_formatted = pd.concat([df_fmt, df_formatted])
    output = df_formatted.to_csv(sep="|", index=False)
    display(Markdown(output))
    with open(filename, 'w') as outfile:
        outfile.write(output)
        
dataset = '../input/survey_results_public.csv'

columns = ['Employment', 'FormalEducation', 'UndergradMajor', 'JobSatisfaction',  
            'HopeFiveYears', 'YearsCodingProf', 'CompanySize', 'YearsCoding',
            'LastNewJob', 'ConvertedSalary', 'EducationTypes', 'LanguageWorkedWith',    
            'IDE', 'OperatingSystem', 'Methodology', 'CheckInCode', 
            'EthicsChoice', 'EthicsReport', 'EthicsResponsible', 'WakeTime', 'SkipMeals',
            'Exercise', 'Gender', 'Age', 'Dependents']

df = pd.read_csv(dataset,
                 dtype='str',
                 na_values=['NA'],
                 usecols=columns)

# Remove any rows without a JobSatisfaction value
df = df[df['JobSatisfaction'].isnull() == False]

df_encoded = pd.get_dummies(df)

frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.3)

# pull the results in a new dataframe to display
results = pd.DataFrame()
result_cols = ['antecedants', 'consequents', 'lift', 'support', 'confidence']
results = rules[result_cols]

# filter the results on rules only containing consquents with 'JobSatisfaction'
results = results[results['consequents'].apply(str).str.contains('JobSatisfaction')]

# sort the results by lift
results = results.sort_values(by='lift', ascending=False)

# filter the results on rules that don't contain 'JobSatisfaction' as an antecedant
results = results[results['antecedants'].apply(str).str.contains('JobSatisfaction') == False]

print('done')
es_results = results[results['consequents'].apply(str).str.contains('JobSatisfaction_Extremely satisfied')]
es_results = es_results.sort_values(by='lift', ascending=False)

# output the top 10 rows
write_markdown('es_rules.md', es_results.head(n=10))
# breakdown of ethics choice per job satisfaction group

data = df[['EthicsChoice', 'JobSatisfaction']].dropna()

# sort the job satisfaction levels from highest to lowest
repl_values = {
    'Extremely satisfied': '1. Extremely satisfied',
    'Moderately satisfied': '2. Moderately satisfied',
    'Slightly satisfied': '3. Slightly satisfied',
    'Neither satisfied nor dissatisfied': '4. Neither satisfied nor dissatisfied',
    'Slightly dissatisfied': '5. Slightly dissatisfied',
    'Moderately dissatisfied': '6. Moderately dissatisfied',
    'Extremely dissatisfied': '7. Extremely dissatisfied'
} 
data['JobSatisfaction'].replace(repl_values, inplace=True)

data_grouped = data.groupby(['JobSatisfaction', 'EthicsChoice']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='EthicsChoice')

g = pivot.plot(kind='bar', width=.8)
g.legend(['Depends', 'No', 'Yes'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_no'] = round((pivot['count', 'No'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_yes'] = round((pivot['count', 'Yes'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_dep'] = round((pivot['count', 'Depends on what it is'] / pivot['count', 'total']) * 100, 2)

pivot
#Checking in code multiple times per day
data = df[['CheckInCode', 'JobSatisfaction']].dropna()

# sort the job satisfaction levels from highest to lowest
data['JobSatisfaction'].replace(repl_values, inplace=True)

data_grouped = data.groupby(['JobSatisfaction', 'CheckInCode']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='CheckInCode')

g = pivot.plot(kind='bar', width=.8)
g.legend(['A few times per week', 'Less than once per month', 'Multiple times per day', 
              'Never', 'Once a day', 'Weekly or a few times per month'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_few_wk'] = round((pivot['count', 'A few times per week'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_monthly'] = round((pivot['count', 'Less than once per month'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_mtl_day'] = round((pivot['count', 'Multiple times per day'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_never'] = round((pivot['count', 'Never'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_daily'] = round((pivot['count', 'Once a day'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_wkly'] = round((pivot['count', 'Weekly or a few times per month'] / pivot['count', 'total']) * 100, 2)

pivot

data = df[['Gender', 'JobSatisfaction']].dropna()
data['JobSatisfaction'].replace(repl_values, inplace=True)
data['Gender'] = data['Gender'].apply(lambda x: 'Female' if 'Female' in x else ('Male' if 'Male' in x else 'Other'))
data_grouped = data.groupby(['JobSatisfaction', 'Gender']).size().reset_index(name='count')

pivot = data_grouped.pivot(index='JobSatisfaction', columns='Gender')

g = pivot.plot(kind='bar', width=.8)
g.legend(['Female', 'Male', 'Other'])

pivot['count', 'total'] = pivot.sum(level=0, axis=1)
pivot['count', 'pct_Female'] = round((pivot['count', 'Female'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_Male'] = round((pivot['count', 'Male'] / pivot['count', 'total']) * 100, 2)
pivot['count', 'pct_Other'] = round((pivot['count', 'Other'] / pivot['count', 'total']) * 100, 2)

pivot
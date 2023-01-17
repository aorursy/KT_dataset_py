import sys
import os
import pandas as pd
import numpy as np

### Import swat
import swat
###

# Set Graphing Options
from matplotlib import pyplot as plt
%matplotlib inline

# Set column/row display options to be unlimited
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#Connect to CAS
s = swat.CAS(os.environ['CASHOST'], os.environ['CASPORT'], None, os.environ.get("SAS_VIYA_TOKEN"))
s
s.sessionProp.setsessopt(caslib='DLUS34')

## We can also enable session metrics to view how long our procedures take to run
s.session.metrics(on=True)
# Load actionsets for analysis (for data prep, modelling, assessing)
actionsets = ['cardinality', 'sampling', 'decisionTree']
[s.loadactionset(i) for i in actionsets]
s.help(actionSet='cardinality')
table_name = "looking_glass_v4"

castbl = s.load_path(
  table_name+'.sas7bdat',
  casOut=dict(name=table_name, replace=True)
)
castbl.head()
# Create table of summary statistics in SAS
castbl.cardinality.summarize(
    cardinality=dict(name = 'full_data_card', replace = True)
)
full_data_card = s.CASTable('full_data_card').to_frame() # bring the summary data locally

# Modify SAS output table using Python to present summary statistics
full_data_card['_PCTMISS_'] = (full_data_card['_NMISS_']/full_data_card['_NOBS_'])*100
print('\n', 'Summary Statistics'.center(90, ' '))
full_data_card[['_VARNAME_','_TYPE_','_PCTMISS_','_MIN_','_MAX_','_MEAN_','_STDDEV_','_SKEWNESS_','_KURTOSIS_']].round(2)
'''Note, you can set the following option to fetch more rows of data out Viya memory - this defaults to 10000 rows.'''
# swat.options.cas.dataset.max_rows_fetched=60000
select_castbl.hist(figsize = (15, 10));
#Declare input variables
target =  'upsell_xsell'
input_vars = ['count_of_suspensions_6m', 'avg_days_susp', 'curr_days_susp', 'calls_in_pk', 'bill_data_usg_m02']
variables = [target] + input_vars
              
select_castbl = castbl[variables]
select_castbl.head(20)
# Create table of summary statistics in SAS
select_castbl.cardinality.summarize(
    varList=[
        {'vars': input_vars}
    ],
    cardinality=dict(name = 'data_card', replace = True)
)

df_data_card = s.CASTable('data_card').to_frame() # bring the summary data locally

# Modify SAS output table using Python to present summary statistics
df_data_card['_PCTMISS_'] = (df_data_card['_NMISS_']/df_data_card['_NOBS_'])*100
print('\n', 'Summary Statistics'.center(90, ' '))
df_data_card[['_VARNAME_','_TYPE_','_PCTMISS_','_MIN_','_MAX_','_MEAN_','_STDDEV_','_SKEWNESS_','_KURTOSIS_']].round(2)
## Note, you can set the following option to fetch more rows of data out Viya memory - this defaults to 10000 rows.
swat.options.cas.dataset.max_rows_fetched=60000
select_castbl.hist(figsize = (15, 10));
s.dataPreprocess.impute(
    table = select_castbl,
    outVarsNamePrefix = 'IMP',
    methodContinuous  = 'MEDIAN',
    inputs            = 'bill_data_usg_m02',
    copyAllVars       = True,
    casOut            = dict(caslib= 'DLUS34', name=table_name, replace=True)
)

# Print the first five rows with imputations
imp_input_vars = ['IMP_bill_data_usg_m02']

total_inputs = input_vars + imp_input_vars
total_inputs.remove('bill_data_usg_m02')

select_castbl = s.CASTable(table_name)[total_inputs]
select_castbl.head(5)
# Create a 70/30 simple random sample split
select_castbl.sampling.srs(
    samppct = 70,
    partind = True,
    seed    = 1,
    output  = dict(
        casOut = dict(
            name=f'{table_name}',
            replace=True), 
        copyVars = 'ALL'
    ),
    outputTables=dict(replace=True)
)
# Train Decision Tree model
s.decisionTree.dtreeTrain(
    table = dict(name= table_name, where='_partind_ = 1'),
    target = target,
    inputs = total_inputs,
    nominals = target, # Encode target as a category, since this is a binary classifcation problem,
    casOut = dict(name='DT_model', replace=True),
    code = dict(casout=dict(name='DT_model_code', replace=True)),
    encodeName=True
)
s.CASTable('DT_model').head()
score_model = s.decisionTree.dtreeScore(
    encodeName = True,
    table      = dict(name = table_name, where='_partind_=0'),
    modelTable = 'DT_model',
    copyVars   = [target, '_partind_'],
    casOut     = dict(name = '_scored_DT', replace = True)
)
score_model
# Load into memory 
castbls = s.load_path(
  'KAGGLETEST_LOOKING_GLASS_1_V3.sas7bdat',
  caslib='ACADCOMP',
  casOut=dict(name='testset', replace=True)
)

# Impute evaluation data
s.dataPreprocess.impute(
    table = castbls,
    outVarsNamePrefix = 'IMP',
    methodContinuous  = 'MEDIAN',
    inputs            = 'bill_data_usg_m02',
    copyAllVars       = True,
    casOut            = dict(caslib= 'DLUS34', name='testset', replace=True)
)

# Score
eval_model = s.decisionTree.dtreeScore(
    encodeName = True,
    table      = 'testset',
    modelTable = 'DT_model',
    copyVars   = ['Customer_ID'],
    casOut     = dict(name = '_eval_DT', replace = True)
)
eval_model
keepcolumns = ['I_upsell_xsell']

evaluation = s.CASTable('_eval_DT').loc[:,keepcolumns]
evaluation.head()
## Output column as CSV - make sure you set the data download limit to be greater than 10000 rows since the test set has more than 10k samples.
swat.options.cas.dataset.max_rows_fetched=60000
evaluation.to_csv('predictionColumn_Python.csv', index=False, float_format='%.12g')
# Set the option to display the entire text and then save to a physical file
pd.set_option("display.max_colwidth", -1)
file=open('DT_score.sas', 'w') 
file.write(s.CASTable('DT_model_code')['DataStepSrc'].str.strip().to_string(index=False).replace("\\n",""))
file.close()
'''To view the score code, run this cell'''
#s.CASTable('DT_model_code')['DataStepSrc'].str.strip().to_string(index=False).replace("\\n","")
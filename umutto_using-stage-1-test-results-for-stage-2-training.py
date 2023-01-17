import pandas as pd 
df_variants_train = pd.read_csv('../input/training_variants', usecols=['Gene', 'Variation', 'Class'])

df_text_train = pd.read_csv('../input/training_text', sep='\|\|', engine='python', 

                            skiprows=1, names=['ID', 'Text'])

df_variants_train['Text'] = df_text_train['Text']

df_train = df_variants_train
df_variants_test = pd.read_csv('../input/test_variants', usecols=['ID', 'Gene', 'Variation'])

df_text_test = pd.read_csv('../input/test_text', sep='\|\|', engine='python', 

                           skiprows=1, names=['ID', 'Text'])

df_variants_test['Text'] = df_text_test['Text']

df_test = df_variants_test
# read stage1 solutions

df_labels_test = pd.read_csv('../input/stage1_solution_filtered.csv')

df_labels_test['Class'] = pd.to_numeric(df_labels_test.drop('ID', axis=1).idxmax(axis=1).str[5:])



# join with test_data on same indexes

df_test = df_test.merge(df_labels_test[['ID', 'Class']], on='ID', how='left').drop('ID', axis=1)

df_test = df_test[df_test['Class'].notnull()]



# join train and test files

df_stage_2_train = pd.concat([df_train, df_test])
df_stage_2_train.reset_index(drop=True, inplace=True)
df_stage_2_train.info()
pd.concat([df_test['Class'].value_counts(), df_stage_2_train['Class'].value_counts()], 

          axis=1, keys=['Test', 'All'])
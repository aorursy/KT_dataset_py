import pandas as pd
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
sample_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
test_df
for i in range(len(test_df)):

    sample_df.loc[sample_df['Patient_Week'].str.contains(test_df.Patient[i]), 'FVC'] = test_df.FVC[i]
sample_df
sample_df.to_csv('submission.csv', index=False)
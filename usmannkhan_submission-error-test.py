import pandas as pd

sample_sub_df = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')

sample_sub_df.to_csv('submission.csv')
sample_sub_df
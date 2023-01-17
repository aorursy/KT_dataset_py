import pandas as pd



train = pd.read_csv('../input/gl-hack-landmarks/train.csv')

sample_submission = pd.read_csv('../input/gl-hack-landmarks/sample_submission.csv')



coord_list = list(sample_submission.columns)[1:]



for c in coord_list:

    sample_submission[c] = train[c].mean()

    

sample_submission.to_csv('submission.csv', index=False)
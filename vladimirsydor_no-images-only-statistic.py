import pandas as pd



train = pd.read_csv('../input/gl-hack-eye-tracking/train.csv')

sample_submission = pd.read_csv('../input/gl-hack-eye-tracking/sample_submission.csv')



sample_submission['x'] = train['x'].mean()

sample_submission['y'] = train['y'].mean()



sample_submission.to_csv('submission.csv', index=False)
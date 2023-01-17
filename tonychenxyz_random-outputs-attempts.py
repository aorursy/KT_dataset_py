import pandas as pd 

from random import uniform as rdm
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")

submission['label'] = submission['label'].apply(lambda x: rdm(0.47, 0.53))

submission.to_csv('submission.csv', index=False)
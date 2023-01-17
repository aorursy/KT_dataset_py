import pandas as pd
!pwd
result = pd.read_csv('../input/gruresult/kernel_submission.csv')
result.head()
result['price'] =result['price'].apply(abs)
result['test_id'] = result['test_id'].astype(int)
result.to_csv('submission_gru_1230.csv',index=False)
result['price'].describe()
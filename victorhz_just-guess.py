import pandas as pd 
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train_human_labels.csv')
df.head()
l = df.LabelName.value_counts().index.tolist()[:4]
an = ' '.join(l)
print(l)
print(an)
df = pd.read_csv('../input/stage_1_sample_submission.csv', usecols=[0])
ans = []
for i in range(len(df)):
    ans.append(an)
df['labels'] = ans
df.head()
df.to_csv('sub.csv', index=False)
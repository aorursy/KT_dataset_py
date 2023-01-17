import pandas as pd

import math
df = pd.read_csv('/kaggle/input/human-protein-atlas-image-classification/train.csv')

df.head()
def get_target_list(str_target):

    return [int(i) for i in str_target.split()]



def label_in_target_list(label, target_list):

    if label in target_list:

        return 1

    return 0
LABELS = list(range(28))
def train_test_split(df, size=0.1, random_seed=8430):

    """

    df: pd.DataFame with columns ['Id', 'Target']

    size: fraction for test set

    random_seed:

    

    return: train and test datafame 

    """

    

    df['Target_List'] = df['Target'].apply(lambda x: get_target_list(x))



    for label in LABELS:

        df[label] = df['Target_List'].apply(lambda tl: label_in_target_list(label, tl))

    

    sorted_counts_by_label = df[LABELS].sum().sort_values()



    val_size = (sorted_counts_by_label * size).apply(lambda x: math.ceil(x))

    

    df['Test_Set'] = 0



    counter = pd.Series(index=LABELS, data=0)

    for label, total in val_size.items():

        num_to_sample = total - counter.loc[label]

        idx = df[(df['Test_Set'] == 0) & (df[label] == 1)].sample(

            num_to_sample, random_state=random_seed).index

        counter += df.loc[idx][LABELS].sum()

        df.at[idx, 'Test_Set'] = 1



    return df[df['Test_Set']==0][['Id', 'Target']], df[df['Test_Set']==1][['Id', 'Target']]
train_df, test_df = train_test_split(df, size=0.1, random_seed=3561)
train_df.to_csv('train_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)
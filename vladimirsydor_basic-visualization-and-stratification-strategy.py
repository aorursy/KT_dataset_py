from typing import Mapping



import pandas as pd

import numpy as np

import seaborn as sns



from functools import reduce

from sklearn.model_selection import StratifiedKFold

from matplotlib import pyplot as plt



%matplotlib inline
path = '../input/stanford-covid-vaccine/'

train = pd.read_json(f'{path}/train.json',lines=True).drop(columns='index')

test = pd.read_json(f'{path}/test.json', lines=True).drop(columns='index')

sub = pd.read_csv(f'{path}/sample_submission.csv')
print(f"Train columns: \n{train.columns}")
print(f"Test columns: \n{test.columns}")
print('=======================Train========================')

sns.countplot(train.seq_length)

plt.show()



sns.countplot(train.seq_scored)

plt.show()



sns.countplot(train.SN_filter)

plt.show()



plt.title('signal_to_noise')

train['signal_to_noise'].hist()

plt.show()
print('=======================Test========================')

sns.countplot(test.seq_length)

plt.show()



sns.countplot(test.seq_scored)

plt.show()
def apply_seq2id(

    seq: str,

    a_seq2id: Mapping[str,int]

):

    return np.array([a_seq2id[el] for el in seq])
STRUCTURE_CODE = {

    '(': 0, 

    '.': 1, 

    ')': 2

}

STRUCTURE_CODE
PREDICTED_LOOP_TYPE_CODE = {

    'H': 0, 

    'E': 1, 

    'B': 2, 

    'M': 3, 

    'X': 4, 

    'S': 5, 

    'I': 6

}

PREDICTED_LOOP_TYPE_CODE
SEQUANCE_CODE = {

    'U': 0, 

    'C': 1, 

    'A': 2, 

    'G': 3

}

SEQUANCE_CODE
print('=======================Train========================')

idx = 0



target_len = train.iloc[idx].seq_scored



plt.title('sequence')

plt.plot(apply_seq2id(train.iloc[idx].sequence, SEQUANCE_CODE)[:target_len])

plt.show()



plt.title('predicted_loop_type')

plt.plot(apply_seq2id(train.iloc[idx].predicted_loop_type, PREDICTED_LOOP_TYPE_CODE)[:target_len])

plt.show()



plt.title('structure')

plt.plot(apply_seq2id(train.iloc[idx].structure, STRUCTURE_CODE)[:target_len])

plt.show()



plt.title('reactivity')

plt.plot(train.iloc[idx].reactivity[:target_len])

plt.show()



plt.title('deg_Mg_pH10')

plt.plot(train.iloc[idx].deg_Mg_pH10[:target_len])

plt.show()



plt.title('deg_Mg_50C')

plt.plot(train.iloc[idx].deg_Mg_50C[:target_len])

plt.show()



plt.title('deg_50C')

plt.plot(train.iloc[idx].deg_50C[:target_len])

plt.show()



plt.title('reactivity_error')

plt.plot(train.iloc[idx].reactivity_error[:target_len])

plt.show()



plt.title('deg_error_Mg_pH10')

plt.plot(train.iloc[idx].deg_error_Mg_pH10[:target_len])

plt.show()



plt.title('deg_error_pH10')

plt.plot(train.iloc[idx].deg_error_pH10[:target_len])

plt.show()



plt.title('deg_error_Mg_50C')

plt.plot(train.iloc[idx].deg_error_Mg_50C[:target_len])

plt.show()



plt.title('deg_error_50C')

plt.plot(train.iloc[idx].deg_error_50C[:target_len])

plt.show()
print('=======================Test========================')

idx = 0



target_len = test.iloc[idx].seq_scored



plt.title('sequence')

plt.plot(apply_seq2id(test.iloc[idx].sequence, SEQUANCE_CODE)[:target_len])

plt.show()



plt.title('predicted_loop_type')

plt.plot(apply_seq2id(test.iloc[idx].predicted_loop_type, PREDICTED_LOOP_TYPE_CODE)[:target_len])

plt.show()



plt.title('structure')

plt.plot(apply_seq2id(test.iloc[idx].structure, STRUCTURE_CODE)[:target_len])

plt.show()
all_structures = reduce(lambda x,y: x+y, train['structure'].apply(list).tolist())

all_sequences = reduce(lambda x,y: x+y, train['sequence'].apply(list).tolist())

all_predicted_loop_types = reduce(lambda x,y: x+y, train['predicted_loop_type'].apply(list).tolist())

all_reactivitys = reduce(lambda x,y: x+y, train['reactivity'].apply(list).tolist())

all_deg_Mg_pH10s = reduce(lambda x,y: x+y, train['deg_Mg_pH10'].apply(list).tolist())

all_deg_Mg_50Cs = reduce(lambda x,y: x+y, train['deg_Mg_50C'].apply(list).tolist())

all_deg_50Cs = reduce(lambda x,y: x+y, train['deg_50C'].apply(list).tolist())

all_deg_pH10s = reduce(lambda x,y: x+y, train['deg_pH10'].apply(list).tolist())
print('=======================Train========================')



plt.title('Structure distribution')

sns.countplot(all_structures)

plt.show()



plt.title('sequence distribution')

sns.countplot(all_sequences)

plt.show()



plt.title('predcited loop types distribution')

sns.countplot(all_predicted_loop_types)

plt.show()



plt.title('reactivity distribution')

plt.hist(all_reactivitys)

plt.show()



plt.title('deg_Mg_pH10 distribution')

plt.hist(all_deg_Mg_pH10s)

plt.show()



plt.title('deg_Mg_50C distribution')

plt.hist(all_deg_Mg_50Cs)

plt.show()



plt.title('deg_50C distribution')

plt.hist(all_deg_50Cs)

plt.show()



plt.title('deg_pH10 distribution')

plt.hist(all_deg_pH10s)

plt.show()
all_structures = reduce(lambda x,y: x+y, test['structure'].apply(list).tolist())

all_sequences = reduce(lambda x,y: x+y, test['sequence'].apply(list).tolist())

all_predicted_loop_types = reduce(lambda x,y: x+y, test['predicted_loop_type'].apply(list).tolist())
print('=======================Test========================')

plt.title('Structure distribution')

sns.countplot(all_structures)

plt.show()



plt.title('sequence distribution')

sns.countplot(all_sequences)

plt.show()



plt.title('predcited loop types distribution')

sns.countplot(all_predicted_loop_types)

plt.show()
median_raw_reactivity = train['reactivity'].apply(np.median)

median_raw_deg_Mg_pH10 = train['deg_Mg_pH10'].apply(np.median)

median_raw_deg_Mg_50C = train['deg_Mg_50C'].apply(np.median)

median_raw_deg_50C = train['deg_50C'].apply(np.median)

median_raw_deg_pH10 = train['deg_pH10'].apply(np.median)
print('=======================Train========================')



plt.title('Median raw reactivity distribution')

median_raw_reactivity.hist()

plt.show()



plt.title('Median raw deg_Mg_pH10 distribution')

median_raw_deg_Mg_pH10.hist()

plt.show()



plt.title('Median raw deg_Mg_50C distribution')

median_raw_deg_Mg_50C.hist()

plt.show()



plt.title('Median raw deg_50C distribution')

median_raw_deg_50C.hist()

plt.show()



plt.title('Median raw deg_pH10 distribution')

median_raw_deg_pH10.hist()

plt.show()
stratify_col = 'reactivity'

bins = 20



train[f'median_raw_{stratify_col}'] = train[stratify_col].apply(np.median)



bins = np.linspace(

    train[f'median_raw_{stratify_col}'].min(), 

    train[f'median_raw_{stratify_col}'].max(), 

    bins

)

train[f'median_raw_bins_{stratify_col}'] = np.digitize(

    train[f'median_raw_{stratify_col}'],

    bins

)



final_stratify_col = f'median_raw_bins_{stratify_col}'
skf = StratifiedKFold(

    n_splits=5,

    random_state=42,

    shuffle=True

)



print("N folds : "+str(skf.get_n_splits(train, train[final_stratify_col])))



fold_indices = list(skf.split(train, train[final_stratify_col]))
for idx, (train_indices, test_indices) in enumerate(fold_indices):

    train_fold = train.iloc[train_indices]

    test_fold = train.iloc[test_indices]

    

    print(f'========Fold {idx+1}=========')

    

    print(f'Train len: {train_fold.shape[0]}')

    print(f'Test len: {test_fold.shape[0]}')

    

    plt.title(f'Train median raw {stratify_col} distribution')

    plt.hist(train_fold[stratify_col].apply(np.median))

    plt.show()

    

    plt.title(f'Test median raw {stratify_col} distribution')

    plt.hist(train_fold[stratify_col].apply(np.median))

    plt.show()

    

    plt.title(f'Train median raw deg_Mg_pH10 distribution')

    plt.hist(train_fold['deg_Mg_pH10'].apply(np.median))

    plt.show()

    

    plt.title(f'Test median raw deg_Mg_pH10 distribution')

    plt.hist(train_fold['deg_Mg_pH10'].apply(np.median))

    plt.show()

    

    plt.title(f'Train median raw deg_pH10 distribution')

    plt.hist(train_fold['deg_pH10'].apply(np.median))

    plt.show()

    

    plt.title(f'Test median raw deg_pH10 distribution')

    plt.hist(train_fold['deg_pH10'].apply(np.median))

    plt.show()

    

    plt.title(f'Train median raw deg_Mg_50C distribution')

    plt.hist(train_fold['deg_Mg_50C'].apply(np.median))

    plt.show()

    

    plt.title(f'Test median raw deg_Mg_50C distribution')

    plt.hist(train_fold['deg_Mg_50C'].apply(np.median))

    plt.show()

    

    plt.title(f'Train median raw deg_50C distribution')

    plt.hist(train_fold['deg_50C'].apply(np.median))

    plt.show()

    

    plt.title(f'Test median raw deg_50C distribution')

    plt.hist(train_fold['deg_50C'].apply(np.median))

    plt.show()
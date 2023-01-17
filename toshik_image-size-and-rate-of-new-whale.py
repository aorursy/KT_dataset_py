import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
def get_size_list(targets, dir_target):

    result = list()

    for target in tqdm(targets):

        img = np.array(Image.open(os.path.join(dir_target, target)))
        result.append(str(img.shape))

    return result
data = pd.read_csv('../input/train.csv')
data['size_info'] = get_size_list(data.Image.tolist(), dir_target='../input/train')
data.to_csv('./size_train.csv', index=False)
counts = data.size_info.value_counts()

agg = data.groupby('size_info').Id.agg({'number_sample': len,
                                        'rate_new_whale': lambda g: np.mean(g == 'new_whale')})

agg = agg.sort_values('number_sample', ascending=False)
agg.to_csv('result.csv')
print(agg.head(20))

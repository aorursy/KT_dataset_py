import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm
import gensim
with open('vk_music_lists.csv', 'w') as f:
    for chunk in pd.read_csv('../input/vk_dataset_anon.csv', 
                                       iterator=True, 
                                       chunksize=1000000,
                                       header=None,
                                       lineterminator='\n',
                                       error_bad_lines=False,
                                       # remove this to use whole dataset
                                       nrows=5000000,
                                       ##################################
                                      ):
        chunk.columns = ['user', 'song', 'band']
        chunk['band'] = chunk['band'].astype(str)
        chunk = chunk.groupby('user')['band'].apply(','.join).reset_index()

        for row in chunk.iterrows():
            f.write(row[1]['band'] + '\n')
class TextToW2V(object):
    def __init__(self, file_path):
        self.file_path = file_path


    def __iter__(self):
        for line in open(self.file_path, 'r'):
            yield line.lower().split(',')[::-1]  # reverse order (make old -> new)

music_collections = TextToW2V('vk_music_lists.csv')
%%time
estimator = gensim.models.Word2Vec(music_collections,
                                   window=15,
                                   min_count=30,
                                   sg=1,
                                   workers=4,
                                   iter=10,
                                   ns_exponent=0.8,
                                  )
user_music = ['led zeppelin', 'Nirvana', 'Pink Floyd']
user_music = [m.lower().strip() for m in user_music]
predicted = estimator.predict_output_word(user_music)

[a[0] for a in predicted if a[0] not in user_music]

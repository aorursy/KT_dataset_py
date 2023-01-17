!pip install /kaggle/input/fastparquet/python_snappy-0.5.4-cp36-cp36m-linux_x86_64.whl

!pip install /kaggle/input/fastparquet/thrift-0.13.0-cp36-cp36m-linux_x86_64.whl

!pip install /kaggle/input/fastparquet/fastparquet-0.3.2-cp36-cp36m-linux_x86_64.whl
import time

from glob import glob



import pandas as pd

from tqdm.auto import tqdm
predictions = []

for path in tqdm(sorted(glob('/kaggle/input/bengaliai-cv19/test_image_data_*.parquet'))):

    tic = time.time()

    data = pd.read_parquet(path, engine='fastparquet')

    toc = time.time()

    print(f'File {path} loaded in {toc-tic:.2f} seconds')

    

    for image_name in data.image_id:

        predictions.append([f'{image_name}_consonant_diacritic', 4])

        predictions.append([f'{image_name}_grapheme_root', 4])

        predictions.append([f'{image_name}_vowel_diacritic', 4])

        

submission = pd.DataFrame(predictions, columns=['row_id', 'target'])

keys = sorted(submission.row_id, key=lambda x : int(x.split('_')[1]))

submission.index = submission.row_id

submission = submission.loc[keys]

submission.to_csv('submission.csv', index=False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import json
import os
import re
os.listdir('/kaggle/input/who-clinical-trials-ictrp-dataset')
root_path = '/kaggle/input/CORD-19-research-challenge'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})
meta_df.head()
# all of the clinical trials data from WHO
clinical_trials_data = pd.read_pickle('/kaggle/input/who-clinical-trials-ictrp-dataset/ICTRP_all_13Apr.pkl')
print(clinical_trials_data.shape)
clinical_trials_data.head()
pd.to_datetime(clinical_trials_data.dt_last_updated).max()
# clinical_trials_data.date_registration.max(),
# clinical_trials_data.date_registration.min()
pd.to_datetime(clinical_trials_data.dt_last_updated).max().month

test_filepath = ('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/7db22f7f81977109d493a0edf8ed75562648e839.json')
with open(test_filepath, 'r') as f:
    test_obj = json.load(f)
test_obj
test_obj.keys()
test_obj['body_text'][0]
def trial_id_regex(text):
    """
    regex to more or less capture all the trial ids
    """
    reg_nct = 'NCT[0-9]{8}'
    reg_chi = 'ChiCTR[0-9]{10}|ChiCTR-[A-Z]{3,}-[0-9]{8}'
    reg_eu = 'EUCTR[0-9]{4}-[0-9]{6}-[0-9]{2}-[A-Z]{2}'
    reg_ir = 'IRCT[0-9]+N[0-9]{1,2}'
    reg_isrctn = 'ISRCTN[0-9]{8}'
    reg_jprn = 'JPRN-[0-9a-zA-Z]+'
    reg_korea = 'KCT[0-9]{7}'
    reg_tctr = 'TCTR[0-9]{11}'
    reg_actrn = 'ACTRN[0-9]{14}'
    reg_drks = 'DRKS[0-9]{8}'
    reg_nl = 'NL[0-9]{4}|NTR[0-9]+'
    reg_cuban = 'RPCEC[0-9]{8}'
    reg_india = 'CTRI\/[0-9]{4}\/[0-9]+\/[0-9]{6}'
    reg_pan_african = 'PACTR[0-9]{15}'
    reg_rbr = 'RBR-[0-9a-z]{6}'
    reg_lbctr ='LBCTR[0-9]{10}'
    reg_per = 'PER-[0-9]{3}-[0-9]{2}'
    reg_slctr = 'SLCTR/[0-9]{4}/[0-9]+'

    registries = [reg_nct, reg_cuban,reg_pan_african, reg_india, reg_pan_african,reg_lbctr,reg_rbr,reg_per,reg_slctr,
                  reg_chi, reg_eu, reg_ir, reg_isrctn, reg_jprn, reg_tctr, reg_actrn, reg_korea, reg_drks, reg_nl]

    reg = ('|').join(registries)
    reg = r'({})'.format(reg)

    return re.findall(reg, text)
def filereader( file_path):
    """
    parse each of the json files, fetching the trial id and any sections etc
    """
    with open(file_path, 'r') as file:
        content = json.load(file)
        paper_id = content['paper_id']
        abstract = []
        body_text = []
        sections = []
        trial_ids = []
        # Abstract
        for entry in content.get('abstract', []):
            abstract.append(entry['text'])
            
        # Body text
        for entry in content.get('body_text', []):
            body_text.append(entry['text'])
            sections.append(entry['section'])
        abstract = '\n'.join(abstract)
        body_text = '\n'.join(body_text)
        trial_ids += trial_id_regex(abstract )
        trial_ids += trial_id_regex(body_text )
        return {'paper_id': paper_id,
                'body_text': body_text, 'abstract': abstract, 'sections':sorted(set(sections)), 'trial_ids': trial_ids}
filereader(test_filepath)
# find all the papers files we need to parse
import glob
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
len(all_json)
from tqdm import tqdm_notebook
dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'trial_ids': [], 'sections': []}
for idx, entry in tqdm_notebook(list(enumerate(all_json))):
    content = filereader(entry)
    dict_['paper_id'].append(content['paper_id'])
    dict_['abstract'].append(content['abstract'])
    dict_['body_text'].append(content['body_text'])
    dict_['sections'].append(content['sections'])
    dict_['trial_ids'].append(content['trial_ids'])
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'sections', 'trial_ids'])
df_covid.to_pickle('cord19-papers-trial_ids.pkl')
df_covid.head()
# find all the papers that mention a clinical trial
papers_with_clinical_trials = df_covid[df_covid.trial_ids.apply(len) >0]
papers_with_clinical_trials.head()
meta_df[meta_df.sha.isin(papers_with_clinical_trials.paper_id)].shape, meta_df[meta_df.pmcid.isin(papers_with_clinical_trials.paper_id)].shape,\
papers_with_clinical_trials.shape
# explode out the list of trial ids so we get 1 row per trial id
trial_id_paper_id_mapping = []
for idx, row in tqdm(list(papers_with_clinical_trials.iterrows())):
    for id in row['trial_ids']:
        trial_id_paper_id_mapping.append({'trial_id': id, 'paper_id': row['paper_id']})
trial_id_paper_id_mapping = pd.DataFrame(trial_id_paper_id_mapping)
trial_id_paper_id_mapping.drop_duplicates(inplace=True)

print(trial_id_paper_id_mapping.shape, 'trial_id_paper_mapping')
trial_data_to_paper_joined = trial_id_paper_id_mapping.merge(ictrp, left_on='trial_id', right_on='trial_id')
trial_data_to_paper_joined = trial_data_to_paper_joined.merge(meta_df, left_on='paper_id', right_on='sha')
trial_data_to_paper_joined.to_pickle(data_dir/'clinical_trial_info_to_paper_info_mapping.zip', compress=3)
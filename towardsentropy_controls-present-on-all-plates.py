import pandas as pd

import os

from pathlib import Path

import numpy as np
path = Path('../input')
def _load_dataset(base_path, dataset, include_controls=True):

    df =  pd.read_csv(os.path.join(base_path, dataset + '.csv'))

    if include_controls:

        controls = pd.read_csv(

            os.path.join(base_path, dataset + '_controls.csv'))

        df['well_type'] = 'treatment'

        df = pd.concat([controls, df], sort=True)

    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])

    df['dataset'] = dataset

    dfs = []

    for site in (1, 2):

        df = df.copy()

        df['site'] = site

        dfs.append(df)

    res = pd.concat(dfs).sort_values(

        by=['id_code', 'site']).set_index('id_code')

    return res
def combine_metadata(base_path=path,

                     include_controls=True):

    df = pd.concat(

        [

            _load_dataset(

                base_path, dataset, include_controls=include_controls)

            for dataset in ['test', 'train']

        ],

        sort=True)

    return df
md = combine_metadata()
neg_ctrls = []

pos_ctrls = []

for experiment in md.experiment.unique():

    for plate in range(1, 5):

        negs = set(md[(md.experiment == experiment) & 

                      (md.plate == plate) & (md.well_type == 'negative_control')].sirna)

        pos = set(md[(md.experiment == experiment) & 

                      (md.plate == plate) & (md.well_type == 'positive_control')].sirna)

        neg_ctrls.append(negs)

        pos_ctrls.append(pos)
positive_controls = set.intersection(*pos_ctrls)

negative_controls = set.intersection(*neg_ctrls)
positive_controls
negative_controls
controls = list(negative_controls) + list(positive_controls)
stats = pd.read_csv(path/'pixel_stats.csv')
md.reset_index(inplace=True)

md['code_site'] = md.apply(lambda row: row['id_code'] + '_' + str(row['site']), axis=1)

stats['code_site'] = stats.apply(lambda row: row['id_code'] + '_' + str(row['site']), axis=1)



merged = pd.merge(md, stats[['mean', 'std', 'median', 'min', 'max', 'code_site']], on='code_site')
control_stats = []

control_ids = []



for experiment in merged.experiment.unique():

    for plate in range(1, 5):

        plate_data = []

        # looping through controls for each plate is slow, but ensures all controls are stored in the same order

        for rna in controls:

            data = merged[(merged.experiment == experiment) &

                          (merged.plate == plate) &

                          (merged.sirna == rna)][['mean', 'std', 'median', 'min', 'max']].values

            

            plate_data.append(data)

            

        data = np.concatenate(plate_data)

        control_stats.append(data)

        control_ids.append(experiment + '_' + str(plate))
control_stats[0].shape
control_ids[0]
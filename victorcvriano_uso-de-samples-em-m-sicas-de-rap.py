import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style = 'whitegrid')
import os

for dirname, _, filenames in os.walk('../input/hiphop-sample-id-dataset/sample_100-master/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_samples = pd.read_csv('../input/hiphop-sample-id-dataset/sample_100-master/samples.csv', encoding = 'iso-8859-1')

df_samples
sns.countplot(x = 'sample_type', palette = 'mako', data = df_samples);
sns.countplot(x = 'interpolation', palette = 'mako', data = df_samples);
df_samples.mean()
sns.lmplot(x = 't_original', y = 't_sample', hue = 'sample_type', palette = 'mako', data = df_samples);
sns.lmplot(x = 't_original', y = 'n_repetitions', data = df_samples);
sns.lmplot(x = 't_sample', y = 'n_repetitions', data = df_samples);
df_tracks = pd.read_csv('../input/hiphop-sample-id-dataset/sample_100-master/tracks.csv',

                        encoding = 'iso-8859-1')



df_tracks
df_samples['original_track_id'].unique()
df_original_tracks = df_tracks[df_tracks['track_id'].isin(['T002', 'T003', 'T005', 'T007', 'T010', 'T011', 'T014', 'T015',

       'T018', 'T021', 'T023', 'T025', 'T027', 'T029', 'T031', 'T033',

       'T035', 'T038', 'T043', 'T040', 'T042', 'T045', 'T049', 'T050',

       'T052', 'T056', 'T058', 'T060', 'T062', 'T064', 'T066', 'T068',

       'T070', 'T072', 'T074', 'T076', 'T078', 'T080', 'T082', 'T083',

       'T085', 'T116', 'T088', 'T090', 'T094', 'T095', 'T098', 'T101',

       'T104', 'T108', 'T111', 'T112', 'T117', 'T119', 'T145', 'T148',

       'T151', 'T154', 'T160', 'T162', 'T167', 'T173', 'T177', 'T180',

       'T182', 'T185', 'T193', 'T199'])]



df_original_tracks
sns.countplot(y = ' genre', palette = 'mako', data = df_original_tracks);
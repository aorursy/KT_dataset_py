!pip install fastai --upgrade >/dev/null
# order of importing the fastai libraries matters here, possibly due to a namespace conflict

from fastai.medical.imaging import *

from fastai.basics import *



import glob
path = Path('../input/rsna-str-pulmonary-embolism-detection')
!ls {path}
path_trn = path/'train'

dirs_trn = path_trn.ls()

dirs_trn[:5].attrgot('name')
path_tst = path/'test'

dirs_tst = path_tst.ls()

print(f'Number of training studies: {len(dirs_trn)}')

print(f'Number of test studies: {len(dirs_tst)}')
fns_trn = L(glob.glob(f'{path_trn}/**/*.dcm', recursive=True))

fns_trn = fns_trn.map(Path)

print(len(fns_trn))

fns_trn[:5]
import gc, os

del(fns_trn)

gc.collect();
fns_trn = L()

for r, d, f in os.walk(path_trn):

    if f:

        fn = Path(f'{r}/{f[0]}')

        fns_trn.append(fn)

print(len(fns_trn))

fns_trn[:5]
fn = fns_trn[0]

dcm = fn.dcmread()

dcm
df_trn = pd.DataFrame.from_dicoms(fns_trn, px_summ=False)

df_trn.to_feather('df_trn.fth')

df_trn.head()
del(df_trn, fns_trn)

gc.collect();
path_lbls = path/'train.csv'

lbls = pd.read_csv(path_lbls)

print(lbls.shape)

lbls.drop_duplicates(['StudyInstanceUID', 'SOPInstanceUID'], inplace=True)

print(lbls.shape)

lbls.head()
lbls.to_feather('lbls.fth')
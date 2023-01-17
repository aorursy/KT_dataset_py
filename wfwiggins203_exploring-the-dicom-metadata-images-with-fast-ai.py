!pip install fastai --upgrade >/dev/null
from fastai.medical.imaging import *

from fastai.basics import *
path_inp = Path('../input')

path = path_inp/'rsna-str-pulmonary-embolism-detection'

path_trn = path/'train'

path_tst = path/'test'
path_df = path_inp/'extracting-dicom-metadata-labels-with-fast-ai'

df_lbls = pd.read_feather(path_df/'lbls.fth')

df_trn = pd.read_feather(path_df/'df_trn.fth')
df_trn.columns
np.random.seed(42)

df_trn.sample(20).T
np.random.seed(42)

fns = L(df_trn.sample(12)['fname'].values.tolist())

dcms = fns.map(dcmread)
!conda install gdcm -c conda-forge -y >/dev/null
import gdcm
for i in range(len(dcms)):

    try:

        dcms[i].show()

    except:

        print('GDCM error')

sops = dcms.map(lambda x: x['SOPInstanceUID'].value)

df_lbls.set_index('SOPInstanceUID').loc[sops]
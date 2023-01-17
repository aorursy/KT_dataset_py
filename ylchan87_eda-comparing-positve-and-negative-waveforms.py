import numpy as np
from scipy import fftpack, signal
import plotly.offline as plt
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import pyarrow.parquet as pq
from tqdm import tqdm
plt.init_notebook_mode(connected=True)
#set_index makes using .loc very handy later on
train_meta = (pd.read_csv('../input/metadata_train.csv')
                .set_index(['id_measurement','phase'])
             )
train_meta.head()
tsum = train_meta.groupby("id_measurement")["target"].sum()
tsum.value_counts()
pos1_id_meas = tsum[tsum==1].index
pos2_id_meas = tsum[tsum==2].index
pos3_id_meas = tsum[tsum==3].index
neg_id_meas = tsum[tsum==0].index[:200] #just pick 200 negative samples for now
train_meta_trimmed = train_meta.loc[pos1_id_meas | pos2_id_meas | pos3_id_meas | neg_id_meas]
xs = pq.read_table('../input/train.parquet', columns=[str(i) for i in train_meta_trimmed['signal_id']]).to_pandas()
#xs = pq.read_table('../input/train.parquet').to_pandas()
print((xs.shape))
xs.columns = [int(c) for c in xs.columns]
def plot_3phase(id_measurement):
    df = train_meta.loc[id_measurement,:]
    print(df)
    sigs = [ xs.loc[:,i] for i in df['signal_id']]
    data = [go.Scattergl( x=xs.index, y=sig) for sig in sigs]
    plt.iplot(data)
pos1_id_meas
plot_3phase(96)
pos2_id_meas
plot_3phase(67)
pos3_id_meas
plot_3phase(1)
neg_id_meas
plot_3phase(0)
def undo_phase(sig):
    """
    find the phase of the 50Hxz component and undo it
    """
    sig_fft = fftpack.fft(sig)
    ang = np.angle(sig_fft[1])
    shift = int((-ang-np.pi/2.)/(2.*np.pi)*len(sig))
    sig_shifted = pd.concat( [sig.iloc[shift:] , sig.iloc[:shift]]).reset_index(drop=True)
    return sig_shifted
for aCol in tqdm(xs.columns):
    xs[aCol] = undo_phase(xs[aCol])
xs_pos = xs[train_meta_trimmed[train_meta_trimmed['target']==1]['signal_id']]
xs_neg = xs[train_meta_trimmed[train_meta_trimmed['target']==0]['signal_id']]
xs_pos_mean = xs_pos.mean(axis=1)
xs_pos_sd = xs_pos.std(axis=1)

xs_neg_mean = xs_neg.mean(axis=1)
xs_neg_sd = xs_neg.std(axis=1)
pos_p1SD = go.Scattergl(
    name='Pos+1SD',
    x=xs.index,
    y=xs_pos_mean + xs_pos_sd,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(color='rgb(55, 0, 0)', width=1),
    )

pos = go.Scatter(
    name='Pos',
    x=xs.index,
    y=xs_pos_mean,
    mode='lines',
    line=dict(color='rgb(255, 0, 0)'),
    )

pos_m1SD = go.Scatter(
    name='Pos-1SD',
    x=xs.index,
    y=xs_pos_mean - xs_pos_sd,
    marker=dict(color="#444"),
    line=dict(color='rgb(55, 0, 0)', width=1),
    mode='lines')

neg_p1SD = go.Scattergl(
    name='Neg+1SD',
    x=xs.index,
    y=xs_neg_mean + xs_neg_sd,
    mode='lines',
    marker=dict(color="#444"),
    line=dict(color='rgb(0, 0, 55)', width=1),
    )

neg = go.Scatter(
    name='Neg',
    x=xs.index,
    y=xs_neg_mean,
    mode='lines',
    line=dict(color='rgb(0, 0, 255)'),
    )

neg_m1SD = go.Scatter(
    name='Neg-1SD',
    x=xs.index,
    y=xs_neg_mean - xs_neg_sd,
    marker=dict(color="#444"),
    line=dict(color='rgb(0, 0, 55)', width=1),
    mode='lines')

# data = [pos_p1SD, pos, pos_m1SD, neg_p1SD, neg, neg_m1SD] #this crash my computer
data = [pos_p1SD, pos, pos_m1SD]
plt.iplot(data)
data = [neg_p1SD, neg, neg_m1SD]
plt.iplot(data)

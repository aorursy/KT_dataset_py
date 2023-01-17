!pip install ase==3.17 schnetpack
import pandas as pd



import torch

import torch.nn.functional as F

from torch.optim import Adam



import schnetpack as spk

import schnetpack.atomistic as atm

import schnetpack.representation as rep

from schnetpack.datasets import *



device = torch.device("cuda")



# load qm9 dataset and download if necessary

data = QM9("qm9/", properties=[QM9.U0], remove_uncharacterized=True)



# Statistics

energies = [data[i][QM9.U0].item() for i in range(len(data))]

energies = pd.Series(energies, name=QM9.U0)

display(energies.describe())

ax = energies.hist(bins=50)

_ = ax.set_xlabel(QM9.U0)
#!rm -r output log



# split in train and val

n_val = 10000

train, val, test = data.create_splits(len(data)-n_val*2, n_val)

loader = spk.data.AtomsLoader(train, batch_size=128, num_workers=2)

val_loader = spk.data.AtomsLoader(val, batch_size=256, num_workers=2)



# create model

reps = rep.SchNet(n_interactions=6)

output = atm.Atomwise()

model = atm.AtomisticModel(reps, output)

model = model.to(device)



# create trainer

max_epochs = 100

opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)

loss = lambda b, p: F.mse_loss(p["y"], b[QM9.U0])

metric_list = [

    spk.metrics.MeanAbsoluteError(QM9.U0, "y"),

    spk.metrics.RootMeanSquaredError(QM9.U0, "y"),

]

hooks = [

    spk.train.MaxEpochHook(max_epochs),

    spk.train.CSVHook('log', metric_list, every_n_epochs=1),

]

trainer = spk.train.Trainer("output/", model, loss,

                            opt, loader, val_loader, hooks=hooks)



# start training

trainer.train(device)
df = pd.read_csv('log/log.csv')

display(df.tail())

_ = df[['MAE_energy_U0','RMSE_energy_U0']].plot(ylim=(0,100))
# This function comes from the following script:

# https://github.com/atomistic-machine-learning/schnetpack/blob/v0.2.1/src/scripts/schnetpack_qm9.py

def evaluate_dataset(metrics, model, loader, device):

    for metric in metrics:

        metric.reset()



    with torch.no_grad():

        for batch in loader:

            batch = {

                k: v.to(device)

                for k, v in batch.items()

            }

            result = model(batch)



            for metric in metrics:

                metric.add_batch(batch, result)



    results = [

        metric.aggregate() for metric in metrics

    ]

    return results
model.load_state_dict(torch.load('output/best_model'))

test_loader = spk.data.AtomsLoader(test, batch_size=256, num_workers=2)

model.eval()



df = pd.DataFrame()

df['metric'] = ['MAE', 'RMSE']

df['training'] = evaluate_dataset(metric_list, model, loader, device)

df['validation'] = evaluate_dataset(metric_list, model, val_loader, device)

df['test'] = evaluate_dataset(metric_list, model, test_loader, device)

df
!ls
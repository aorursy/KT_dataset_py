!pip install ase==3.17 schnetpack
!ls ../input
import numpy as np

import pandas as pd

molecules = pd.read_csv('../input/structures.csv')

molecules = molecules.groupby('molecule_name')

energies = pd.read_csv('../input/potential_energy.csv')

dipoles = pd.read_csv('../input/dipole_moments.csv')

dipoles['scalar'] = np.sqrt(np.square(dipoles[['X', 'Y', 'Z']]).sum(axis=1))
molecules.ngroups
energies.head()
len(energies)
energy_series = pd.Series(energies.set_index('molecule_name')['potential_energy'])

energy_series.describe()
ax = energy_series.hist(bins=50)

_ = ax.set_xlabel("Potential Energy")
dipoles.head()
len(dipoles)
dipole_series = pd.Series(dipoles.set_index('molecule_name')['scalar'])

dipole_series.describe()
ax = dipole_series.hist(bins=50)

_ = ax.set_xlabel("Dipole Moment")
train = pd.read_csv('../input/train.csv')

train.head()
train_molecule_names = train.molecule_name.unique()

len(train_molecule_names)
from ase import Atoms

from ase.db import connect



def create_db(db_path, molecule_names):

    with connect(db_path) as db:

        for name in molecule_names:

            mol = molecules.get_group(name)

            atoms = Atoms(symbols=mol.atom.values,

                          positions=[(row.x,row.y,row.z) for row in mol.itertuples()])

            db.write(atoms, name=name,

                     potential_energy=energy_series.get(name, default=float('nan')),

                     scalar_dipole=dipole_series.get(name, default=float('nan'))

                    )
champs_path = 'CHAMPS_train.db'

dataset_size = len(train_molecule_names) # 20000

dataset_molecule_names = train_molecule_names[:dataset_size]

create_db(db_path=champs_path, molecule_names=dataset_molecule_names)
with connect(champs_path) as db:

    print(len(db))
import schnetpack



properties=['potential_energy', 'scalar_dipole']



dataset = dict()

for p in properties:

    dataset[p] = schnetpack.data.AtomsData(champs_path, properties=[p])
for p in properties:

    print(p, len(dataset[p]))
import pandas as pd



import torch

import torch.nn.functional as F

from torch.optim import Adam



import schnetpack as spk

import schnetpack.atomistic as atm

import schnetpack.representation as rep

from schnetpack.datasets import *



device = torch.device("cuda")
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
def schnet_model(property):

    reps = rep.SchNet(n_interactions=6)

    if 'dipole' in property:

        print('use dipole moment')

        output = atm.DipoleMoment(n_in=128, predict_magnitude=True)

    else:

        output = atm.Atomwise()

    model = atm.AtomisticModel(reps, output)

    model = model.to(device)

    

    return model
def train_model(property, max_epochs=500):

    # split in train and val

    n_dataset = len(dataset[property])

    n_val = n_dataset // 10

    train_data, val_data, test_data = dataset[property].create_splits(n_dataset-n_val*2, n_val)

    train_loader = spk.data.AtomsLoader(train_data, batch_size=128, num_workers=2)

    val_loader = spk.data.AtomsLoader(val_data, batch_size=256, num_workers=2)



    # create model

    model = schnet_model(property)



    # create trainer

    opt = Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)

    loss = lambda b, p: F.mse_loss(p["y"], b[property])

    metrics = [

        spk.metrics.MeanAbsoluteError(property, "y"),

        spk.metrics.RootMeanSquaredError(property, "y"),

    ]

    hooks = [

        spk.train.MaxEpochHook(max_epochs),

        spk.train.CSVHook(property+'/log', metrics, every_n_epochs=1),

    ]

    trainer = spk.train.Trainer(property+'/output', model, loss,

                            opt, train_loader, val_loader, hooks=hooks)



    # start training

    trainer.train(device)

    

    # evaluation

    model.load_state_dict(torch.load(property+'/output/best_model'))

    test_loader = spk.data.AtomsLoader(test_data, batch_size=256, num_workers=2)

    model.eval()



    df = pd.DataFrame()

    df['metric'] = ['MAE', 'RMSE']

    df['training'] = evaluate_dataset(metrics, model, train_loader, device)

    df['validation'] = evaluate_dataset(metrics, model, val_loader, device)

    df['test'] = evaluate_dataset(metrics, model, test_loader, device)

    display(df)

    

    return test_data
def show_history(property):

    df = pd.read_csv(property+'/log/log.csv')

    display(df.tail())

    max_value = None # df['RMSE_'+property].min()*5

    _ = df[['MAE_'+property,'RMSE_'+property]].plot(ylim=(0,max_value))
def test_prediction(dataset, property):

    # create model

    model = schnet_model(property)

    

    # load the best parameters

    model.load_state_dict(torch.load(property+'/output/best_model'))

    loader = spk.data.AtomsLoader(dataset, batch_size=256, num_workers=2)

    model.eval()

    

    # predict molecular properties

    targets = []

    predictions = []

    with torch.no_grad():

        for batch in loader:

            batch = {

                k: v.to(device)

                for k, v in batch.items()

            }

            result = model(batch)

            targets += batch[property].squeeze().tolist()

            predictions += result['y'].squeeze().tolist()

    return targets, predictions
def show_predictions(dataset, property):

    targets, predictions = test_prediction(dataset, property)

    df_pred = pd.DataFrame()

    df_pred['Target'] = targets

    df_pred['Prediction'] = predictions

    df_pred.plot.scatter(x='Target', y='Prediction', title=property)
used_test_data = dict()

for p in properties:

    print(p)

    used_test_data[p] = train_model(p, max_epochs=100)

    show_history(p)
for p in properties:

    show_predictions(used_test_data[p], p)
!mv potential_energy/log/log.csv log_potential.csv

!mv scalar_dipole/log/log.csv log_dipole.csv

!mv potential_energy/output/best_model best_model_potential

!mv scalar_dipole/output/best_model best_model_dipole

!rm -r potential_energy

!rm -r scalar_dipole
!ls
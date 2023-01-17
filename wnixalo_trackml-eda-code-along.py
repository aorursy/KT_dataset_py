%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import seaborn as sns
from trackml.dataset import load_event, load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event
# One event of 8850
event_id = 'event000001000'
# "All methods either take or return pandas.DataFrame objects"
hits, cells, particles, truth = load_event('../input/train_1/' + event_id)
hits.head()
hits.tail()
hits.describe()
# plt.figure(figsize=(10,10))
# plt.scatter(hits.x,hits.y, s=1)
# plt.show()
# Same as above, but include Univariate plots & Pearson correlation coeffs:
radialview = sns.jointplot(hits.x, hits.y, size=10, s=1)
radialview.set_axis_labels('x [mm]', 'y [mm]')
plt.show()
radialview = sns.jointplot(hits[hits.x.abs() < 200].x, hits[hits.y.abs() < 200].y, size=10, s=1)
radialview.set_axis_labels('x [mm]', 'y [mm]');
# plt.show()
def radial_display(dat, lim=2000):
    radialview = sns.jointplot(dat[dat.x.abs()<lim].x, dat[dat.y.abs()<lim].y, size=10, s=1)
    radialview.set_axis_labels('x [mm]', 'y [mm]')
    plt.show()
nocap = hits[hits.z.abs() < 200]
radial_display(nocap, lim=200)
radial_display(nocap, lim=50)
def side_display(fgsz = (24,8)): 
    plt.figure(figsize=fgsz)
    axialview = plt.scatter(hits.z, hits.y, s=1)
    plt.xlabel('z (mm)')
    plt.ylabel('y (mm)')
    plt.show()
side_display()
def iso_display():
    plt.figure(figsize=(15,15))
    ax = plt.axes(projection='3d')
    sample = hits.sample(30000)
    ax.scatter(sample.z, sample.x, sample.y, s=5, alpha=0.5)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('x (mm)')
    ax.set_zlabel('y (mm)')
    # These two added to widen the 3D space
    ax.scatter(3000,3000,3000, s=0)
    ax.scatter(-3000,-3000,-3000, s=0)
    plt.show()
iso_display()
volumes = hits.volume_id.unique()

fg,ax = plt.subplots(figsize=(15,15))
for volume in volumes:
    v = hits[hits.volume_id == volume]
    ax.scatter(v.x, v.y, s=10, label='Volume '+str(volume), alpha=0.5)
ax.set_title('Detector Volumes, Radial View')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.legend()
plt.show()

volumes = hits.volume_id.unique()

fg,ax = plt.subplots(figsize=(24,8))
for volume in volumes:
    v = hits[hits.volume_id == volume]
    ax.scatter(v.z, v.y, s=10, label='Volume '+str(volume), alpha=0.5)
ax.set_title('Detector Volumes, Axial View')
ax.set_xlabel('z [mm]')
ax.set_ylabel('y [mm]')
ax.legend()
plt.show()
sample = hits.sample(30000)
plt.figure(figsize=(20,20))
ax = plt.axes(projection='3d')
for volume in volumes:
    v = sample[sample.volume_id == volume]
    ax.scatter(v.z, v.x, v.y, s=5, label='Volume '+str(volume), alpha=0.5)
ax.set_xlabel('z (mm)'); ax.set_ylabel('x (mm)'); ax.set_zlabel('y (mm)')
ax.legend()
# added to widen the 3D space:
ax.scatter(3000,3000,3000, s=0); ax.scatter(-3000,-3000,-3000, s=0)
plt.show()
# RADIAL
layers = hits.layer_id.unique()
fg,ax  = plt.subplots(figsize=(15,15))
for l_name in layers:
    l = hits[hits.layer_id == l_name]
    ax.scatter(l.x, l.y, s=10, label='Layer '+str(l_name), alpha=0.5)
ax.set_title('Detector Layers, Radial View'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
ax.legend()
plt.show()
# AXIAL
fg,ax  = plt.subplots(figsize=(24,8))
for l_name in layers:
    l = hits[hits.layer_id == l_name]
    ax.scatter(l.z, l.y, s=10, label='Layer '+str(l_name), alpha=0.5)
ax.set_title('Detector Layers, Axial View'); ax.set_xlabel('z [mm]'); ax.set_ylabel('y [mm]')
ax.legend()
plt.show()
# ISOMETRIC
sample = hits.sample(30000)
plt.figure(figsize=(20,20))
ax = plt.axes(projection='3d')
for layer in layers:
    l = sample[sample.layer_id == layer]
    ax.scatter(l.z, l.x, l.y, s=5, label='Layer '+str(layer), alpha=0.5)
ax.set_xlabel('z (mm)'); ax.set_ylabel('x (mm)'); ax.set_zlabel('y (mm)'); ax.legend()
# added to widen the 3D space
ax.scatter(3000,3000,3000, s=0); ax.scatter(-3000,-3000,-3000, s=0)
plt.show()
groups = [hits.volume_id, hits.layer_id, hits.module_id, cells.ch0, cells.ch1]
fig,axes = plt.subplots(1,5, figsize=(30,10))
for i,ax in enumerate(axes):
    sns.distplot(groups[i], ax=ax)
radius2 = np.sqrt(hits.x**2 + hits.y**2)
radius3 = np.sqrt(hits.x**2 + hits.y**2 + hits.z**2)
z2 = hits.z**2
rads = [radius2, radius3, z2]

axlbls = ['sqrt(x^2 + y^2)', 'sqrt(x^2 + y^2 + z^2)', 'z']

fig,axes = plt.subplots(1,3, figsize=(30,10))
for i,ax in enumerate(axes):
    sns.distplot(rads[i], axlabel=axlbls[i], ax=ax)
labels = [['volume_id','radius'],['layer_id','radius'],['module_id','radius']]
groups = [hits.volume_id, hits.layer_id, hits.module_id]
fig,axes = plt.subplots(1,3, figsize=(30,10))
for i,ax in enumerate(axes):
    ax.set_xlabel(labels[i][0]); ax.set_ylabel(labels[i][1])
    ax.scatter(groups[i], radius2)
hits.volume_id.value_counts()
hits.layer_id.value_counts()
hits.module_id.value_counts().head()
# Pairplotting 120k hits takes too long - so a random sample of 3k.
sample = hits.sample(3000)
# Color coding by group
sns.pairplot(sample, hue='volume_id', size=8)
plt.show()
fg,ax = plt.subplots(figsize=(10,10))
hitscorr = hits.drop('hit_id', axis=1).corr()
sns.heatmap(hitscorr, cmap='coolwarm', square=True, ax=ax)
ax.set_title('Hits Correlation Heatmap');
cells.head()
cells.tail()
cells.describe()
fig,axe = plt.subplots(figsize=(10,10))
cellscorr = cells.drop('hit_id', axis=1).corr()
sns.heatmap(cellscorr, cmap='coolwarm', square=True, ax=axe)
axe.set_title('Cells Correlation Heatmap');
particles.head()
particles.tail()
particles.describe()
plt.figure(figsize=(15,10)); 
plt.subplot(1,2,1); plt.xlabel('Charge (e)'); plt.ylabel('Counts')
particles.q.hist(bins=3)
plt.subplot(1,2,2); plt.xlabel('nhits')
particles.nhits.hist(bins=particles.nhits.max())
plt.show();
fig,axe = plt.subplots(figsize=(10,10))
p = np.sqrt(particles.px**2 + particles.py**2 + particles.pz**2)
axe.scatter(particles.nhits, p)
axe.set_yscale('log'); axe.set_xlabel('nhits'); axe.set_ylabel('Momentum [GeV/c]')
plt.show();
fig,axes = plt.subplots(1,2,figsize=(15,8))
axes[0].hist(np.sqrt(particles.px**2 + particles.py**2), bins=100, log=True)
axes[0].set_xlabel('Transverse momentum [GeV/c]'); axes[0].set_ylabel('Counts')
axes[1].hist(particles.pz.abs(), bins=100, log=True)
axes[1].set_xlabel('Z momentum [GeV/c]')
plt.show();
fig,axe = plt.subplots(figsize=(10,10))
axe.scatter(np.sqrt(particles.px**2 + particles.py**2), particles.pz, s=1)
axe.set_xscale('log'); axe.set_xlabel('Transverse momentum [GeV/c]'); axe.set_ylabel('Z momentum [GeV/c]')
plt.show();
p = particles[particles.pz < 200]

fig,axe = plt.subplots(figsize=(10,10))
axe.scatter(np.sqrt(p.px**2 + p.py**2), p.pz, s=3, alpha=0.5)
axe.plot([.1,.1],[p.pz.min(), p.pz.max()], c='g')
axe.plot([.1,np.sqrt(p.px**2 + p.py**2).max()], [.1,.1], c='r', linestyle='--')
axe.set_xscale('log'); axe.set_xlabel('Transverse momentum [GeV/c]'); axe.set_ylabel('Z momentum [GeV/c]')
plt.show();
f,axe = plt.subplots(figsize=(10, 10))
particlescorr = particles.drop('particle_id', axis=1).corr()
sns.heatmap(particlescorr, cmap='coolwarm', square=True, ax = axe)
axe.set_title('Particles Correlation Heatmap')
plt.show();
truth.head()
truth.tail()
# looking at a particle
truth[truth.particle_id == 22525763437723648]
# Number of unique particles
len(truth.particle_id.unique())
# get every kth particle
k = 100; tracks = truth.particle_id.unique()[1::k]

f,axe = plt.subplots(figsize=(15,15))
ax = f.add_subplot(1,1,1, projection='3d')
for track in tracks:
    t = truth[truth.particle_id == track]
    ax.plot3D(t.tz, t.tx, t.ty)
ax.set_xlabel('z [mm]'); ax.set_ylabel('x [mm]'); ax.set_zlabel('y [mm]')
# These two added to widen the 3D space
ax.scatter(3000,3000,3000, s=0); ax.scatter(-3000,-3000,-3000, s=0)
plt.show();







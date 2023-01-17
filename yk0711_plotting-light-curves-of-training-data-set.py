# You can edit the font size here to make rendered text more comfortable to read
# It was built on a 13" retina screen with 18px
from IPython.core.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 18px; }</style>"))

# we'll also use this package to read tables
# it's generally useful for astrophysics work, including this challenge
# so we'd suggest installing it, even if you elect to work with pandas
from astropy.table import Table
import os
import numpy as np
import scipy.stats as spstat
import matplotlib.pyplot as plt
from collections import OrderedDict

%matplotlib inline

from gatspy.periodic import LombScargleMultiband
class LightCurve(object):
    '''Light curve object for PLAsTiCC formatted data'''
    
    _passbands = OrderedDict([(0,'C4'),\
                              (1,'C2'),\
                              (2,'C3'),\
                              (3,'C1'),\
                              (4,'k'),\
                              (5,'C5')])
    
    _pbnames = ['u','g','r','i','z','y']
    
    #def __init__(self, filename):
    def __init__(self, fluxDF):
        '''Read in light curve data'''

        #self.DFlc     = Table.read(filename, format='ascii.csv')
        self.DFlc     = fluxDF
        #self.filename = filename.replace('.csv','')
        self._finalize()
     
    # this is some simple code to demonstrate how to calculate features on these multiband light curves
    # we're not suggesting using these features specifically
    # there also might be additional pre-processing you do before computing anything
    # it's purely for illustration
    def _finalize(self):
        '''Store individual passband fluxes as object attributes'''
        # in this example, we'll use the weighted mean to normalize the features
        weighted_mean = lambda flux, dflux: np.sum(flux*(flux/dflux)**2)/np.sum((flux/dflux)**2)
        
        # define some functions to compute simple descriptive statistics
        normalized_flux_std = lambda flux, wMeanFlux: np.std(flux/wMeanFlux, ddof = 1)
        normalized_amplitude = lambda flux, wMeanFlux: (np.max(flux) - np.min(flux))/wMeanFlux
        normalized_MAD = lambda flux, wMeanFlux: np.median(np.abs((flux - np.median(flux))/wMeanFlux))
        beyond_1std = lambda flux, wMeanFlux: sum(np.abs(flux - wMeanFlux) > np.std(flux, ddof = 1))/len(flux)
        
        for pb in self._passbands:
            ind = self.DFlc['passband'] == pb
            pbname = self._pbnames[pb]
            
            if len(self.DFlc[ind]) == 0:
                setattr(self, f'{pbname}Std', np.nan)
                setattr(self, f'{pbname}Amp', np.nan)
                setattr(self, f'{pbname}MAD', np.nan)
                setattr(self, f'{pbname}Beyond', np.nan)
                setattr(self, f'{pbname}Skew', np.nan)
                continue
            
            f  = self.DFlc['flux'][ind]
            df = self.DFlc['flux_err'][ind]
            m  = weighted_mean(f, df)
            
            # we'll save the measurements in each passband to simplify access.
            setattr(self, f'{pbname}Flux', f)
            setattr(self, f'{pbname}FluxUnc', df)
            setattr(self, f'{pbname}Mean', m)
            
            # compute the features
            std = normalized_flux_std(f, df)
            amp = normalized_amplitude(f, m)
            mad = normalized_MAD(f, m)
            beyond = beyond_1std(f, m)
            skew = spstat.skew(f) 
            
            # and save the features
            setattr(self, f'{pbname}Std', std)
            setattr(self, f'{pbname}Amp', amp)
            setattr(self, f'{pbname}MAD', mad)
            setattr(self, f'{pbname}Beyond', beyond)
            setattr(self, f'{pbname}Skew', skew)
        
        # we can also construct features between passbands
        pbs = list(self._passbands.keys())
        for i, lpb in enumerate(pbs[0:-1]):
            rpb = pbs[i+1]
            
            lpbname = self._pbnames[lpb]
            rpbname = self._pbnames[rpb]
            
            colname = '{}Minus{}'.format(lpbname, rpbname.upper())
            lMean = getattr(self, f'{lpbname}Mean', np.nan)
            rMean = getattr(self, f'{rpbname}Mean', np.nan)
            col = -2.5*np.log10(lMean/rMean) if lMean> 0 and rMean > 0 else -999
            setattr(self, colname, col)
    
    def plot_multicolor_lc(self, target_id=None):
        '''Plot the multiband light curve'''
        
        # Lomb-Scargle
        model = LombScargleMultiband(fit_period=True)
        # we'll window the search range by setting minimums and maximums here
        # but in general, the search range you want to evaluate will depend on the data
        # and you will not be able to window like this unless you know something about
        # the class of the object a priori
        t_min = max(np.median(np.diff(sorted(self.DFlc['mjd']))), 0.1)
        t_max = min(10., (self.DFlc['mjd'].max() - self.DFlc['mjd'].min())/2.)
        
        model.optimizer.set(period_range=(t_min, t_max), first_pass_coverage=5)
        model.fit(self.DFlc['mjd'], self.DFlc['flux'], dy=self.DFlc['flux_err'], filts=self.DFlc['passband'])
        period = model.best_period
        obj_id = self.DFlc['object_id'][0] # object ID
        print(f'object ID: {obj_id} has a period of {period} days')
        
        phase = (self.DFlc['mjd'] /period) % 1
        
        #fig, ax = plt.subplots(figsize=(8,6))
        fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(16,6))

        #if phase is None:
        #    phase = []
        #if len(phase) != len(self.DFlc):
        #    phase = self.DFlc['mjd']
        #    xlabel = 'MJD'
        #else:
        #    xlabel = 'Phase'
          
        for i, pb in enumerate(self._passbands):
            pbname = self._pbnames[pb]
            ind = self.DFlc['passband'] == pb
            if len(self.DFlc[ind]) == 0:
                continue
            # errorbar: plot y versus x as lines and/or markers with attached errorbars
            #ax.errorbar(phase[ind], 
            #         self.DFlc['flux'][ind],
            #         self.DFlc['flux_err'][ind],
            #         fmt = 'o', color = self._passbands[pb], label = f'{pbname}')
            ax.errorbar(self.DFlc['mjd'][ind], 
                     self.DFlc['flux'][ind],
                     self.DFlc['flux_err'][ind],
                     fmt = 'o', color = self._passbands[pb], label = f'{pbname}')
            ax2.errorbar(phase[ind], 
                     self.DFlc['flux'][ind],
                     self.DFlc['flux_err'][ind],
                     fmt = 'o', color = self._passbands[pb], label = f'{pbname}')
        ax.legend(ncol = 4, frameon = True)
        #ax.set_xlabel(f'{xlabel}', fontsize='large')
        ax.set_xlabel('MJD', fontsize='large')
        ax.set_ylabel('Flux', fontsize='large')
        ax2.legend(ncol = 4, frameon = True)
        ax2.set_xlabel('Phase', fontsize='large')
        ax2.set_ylabel('Flux', fontsize='large')
        #fig.suptitle(self.filename, fontsize='x-large')
        fig.suptitle('object ID: ' + str(self.DFlc['object_id'][0]) + ', target ID: ' + str(target_id), fontsize='x-large') # graph title = object ID
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    def get_features(self):
        '''Return all the features for this object'''
        variables = ['Std', 'Amp', 'MAD', 'Beyond', 'Skew']
        feats = []
        for i, pb in enumerate(self._passbands):
            pbname = self._pbnames[pb]
            feats += [getattr(self, f'{pbname}{x}', np.nan) for x in variables]
        return feats
trainfilename = '../input/PLAsTiCC-2018/training_set.csv'
train = Table.read(trainfilename, format='csv')
train
# read a sample object data
obj_id = train['object_id'][0]
fluxDF = train[train['object_id'] == obj_id]
fluxDF
lc = LightCurve(fluxDF)
lc.plot_multicolor_lc()
trainmetafilename = '../input/PLAsTiCC-2018/training_set_metadata.csv'
train_meta = Table.read(trainmetafilename, format='csv')
# set: remove duplicated object_id
# list: change type to list
target_list = list(set(train_meta['target']))
all_obj_id_list = []

# create a list: object IDs which has same target ID
for target_id in target_list:
    #print(train_meta[train_meta['target'] == target_id]['object_id'])
    all_obj_id_list.append(train_meta[train_meta['target'] == target_id]['object_id'])
    
for i in range(len(all_obj_id_list)):
#for i in range(2):
    print('********* target ID: ' + str(target_list[i]) + ' *********')
    each_obj_id_list = all_obj_id_list[i]
    print('********* ' + str(len(each_obj_id_list)) + ' objects in target type ' + str(target_list[i]) + ' *********')
    # WARNING: this takes very long time...
    #for j in range(len(each_obj_id_list)):
    for j in range(10):
        obj_id = each_obj_id_list[j]
        print ('****** object ID: ' + str(obj_id) + ' ******')
        fluxDF = train[train['object_id'] == obj_id]
        lc = LightCurve(fluxDF)
        lc.plot_multicolor_lc(target_list[i])

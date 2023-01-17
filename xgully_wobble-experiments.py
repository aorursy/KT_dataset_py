import wobble
import tensorflow
tensorflow.__version__
#wobble.__file__
#wobble.utils.<TAB>
data = wobble.Data('/kaggle/input/51peg_e2ds.hdf5', orders=[30, 31, 32, 33, 34, 35])
results = wobble.Results(data=data)
#model = wobble.Model(data, results, 0)
import time
%%capture
t0 = time.time()
results = wobble.Results(data=data)
for r in range(len(data.orders)):
        model = wobble.Model(data, results, r)
        model.add_star('star')
        model.add_telluric('tellurics')
        wobble.optimize_order(model)
        
t1 = time.time()
dtime = (t1 - t0) / 60.0
print("It took {:0.1f} minutes to run the model ".format(dtime))
data.xs[0].shape
data.fluxes[0].shape
import numpy as np
import matplotlib.pyplot as plt
r = 0
r+=1
plt.plot(np.exp(results.star_template_xs[r]), np.exp(results.star_template_ys[r]),
                 label='star')
plt.plot(np.exp(results.tellurics_template_xs[r]), np.exp(results.tellurics_template_ys[r]),
                label='tellurics')
plt.plot(np.exp(data.xs[r][0]), np.exp(data.ys[r][0])+1,
                label='data')
plt.xlabel('Wavelength (Ang)')
plt.ylabel('Normalized Flux')
plt.legend()
plt.ylim(0, 3)
plt.show()
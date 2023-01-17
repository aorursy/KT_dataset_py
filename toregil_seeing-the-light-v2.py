import numpy as np # linear algebra

import matplotlib.pyplot as plt # graphs

%matplotlib inline

from scipy.ndimage.filters import uniform_filter # to smooth images



from sklearn.linear_model import LinearRegression



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = np.loadtxt('../input/data141110.csv', delimiter=',', skiprows=1)

image_no=data[:,0].reshape(-1,1)

frame_no=data[:,1].reshape(-1,1)

time_hrs=data[:,2].reshape(-1,1)

nb_images = data.shape[0]
flo_image_1 = np.load('../input/flo_image_1.npz')

flo_image_2 = np.load('../input/flo_image_2.npz')

image_ids = np.concatenate([flo_image_1['image_ids'], flo_image_2['image_ids']])

images = np.concatenate([flo_image_1['image_stack'], flo_image_2['image_stack']])

del flo_image_1, flo_image_2
plt.plot(np.arange(nb_images), time_hrs)
fig = plt.figure(figsize = (12,12))

plt.imshow(images[100], cmap='hot')
mean_values = np.mean(images, axis=(1,2))

plt.plot(time_hrs, mean_values, 'b')
model = LinearRegression()

model.fit(time_hrs, mean_values)

plt.plot(time_hrs, model.predict(time_hrs), 'r')

plt.plot(time_hrs, mean_values, 'b')
mean_linear = model.predict(time_hrs)

norm_values = mean_values - mean_linear

plt.plot(time_hrs, norm_values, 'b')
def remove_linear_trend(time_series):

    x_axis = np.arange(len(time_series)).reshape(-1, 1)

    model = LinearRegression()

    model.fit(x_axis, time_series)

    return time_series - model.predict(x_axis)
spectrum = np.fft.rfft(norm_values)

freq = np.fft.rfftfreq(len(norm_values), 1/10.)  #because there are ten image frames per hour

plt.plot(1.0 / freq[3:], abs(spectrum[3:len(freq)]))
acorr = plt.acorr(norm_values, maxlags=300)
def filtered_acorr(time_series, high_pass=None, unbiased=True):

    """

    high_pass is a short 1D float array, 

    by which the low-frequency amplitudes are multiplied,

    e.g. high_pass = [0,0,0,0,0,0]

    """

    N = len(time_series)

    norm_values = remove_linear_trend(time_series)

    spectrum = np.fft.fft(norm_values, n=2*N)

    if high_pass is not None:

        spectrum[0] *= high_pass[0]

        for i in range(len(high_pass)):

            spectrum[i] *= high_pass[i]

            spectrum[-i] *= high_pass[i]

    acorr = np.real(np.fft.ifft(spectrum * np.conj(spectrum))[:N])

    if unbiased:

        return acorr / (N - np.arange(N))

    else:

        return acorr / N
def get_period(acorr):

    """

    Returns the index with largest acorr value, 

    after the first zero crossing.

    There are of course more sophisticated methods of doing this.

    """

    negative_periods = np.where(acorr <= 0.0)

    if negative_periods[0].size == 0:

        return 0

    first_zero = np.min(negative_periods)

    return first_zero + np.argmax(acorr[first_zero:])
print(get_period(filtered_acorr(mean_values)[:300]))
def get_grid_periods(images, box_size, max_period=300, high_pass=np.zeros((10,)), unbiased=False):

    """

    periods, acorrs = get_grid_periods(images, box_size, max_period=300, 

                                    high_pass=np.zeros((10,)), unbiased=False)

    Divides the image domain into small boxes of size box_size = (h,w) and computes

    the period over each of these small boxes."""

    h,w = box_size

    rows = images.shape[1] // h

    cols = images.shape[2] // w

    acorrs = np.empty((rows, cols, max_period), dtype = "float32")

    periods = np.empty((rows,cols, ), dtype = "int")

    for i in range(rows):

        for j in range(cols):

            time_series = np.mean(images[:, i*h:(i+1)*h, j*w:(j+1)*w], axis=(1,2))

            acorrs[i ,j] = filtered_acorr(time_series, high_pass = high_pass, 

                                          unbiased=unbiased)[:max_period]

            periods[i ,j] = get_period(acorrs[i, j])

    return periods, acorrs
periods, acorrs = get_grid_periods(images, (32, 32), max_period=1000, unbiased=False)

plt.imshow(periods, cmap='hot')

print(periods)
plt.plot(acorrs[13,5], color ='k')

plt.plot(acorrs[3,8], color='r')

plt.plot(acorrs[5,1], color='y')
fix, ax = plt.subplots(1,8, figsize=(12,4))

idx = [80, 200, 320, 440, 560, 740, 860, 980]

for j in range(8):

    ax[j].set_axis_off()

    ax[j].imshow(images[idx[j],:,:] - mean_linear[idx[j]], cmap='hot', vmin=-800, vmax=6400)

        
fig, ax = plt.subplots(1,4,figsize=(12,12))

ax[0].imshow(images[100], cmap='hot', vmin=1500, vmax=6000)

ax[1].imshow(images[320], cmap='hot', vmin=1500, vmax=6000)

ax[2].imshow(images[1050], cmap='hot', vmin=1500, vmax=6000)

ax[3].imshow(images[1530], cmap='hot', vmin=1500, vmax=6000)
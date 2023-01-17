import numpy as np

import math

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from copy import deepcopy

def create_axes_grid(numplots_x, numplots_y, plotsize_x=6, plotsize_y=3):

    fig, axes = plt.subplots(numplots_y, numplots_x)

    fig.set_size_inches(plotsize_x * numplots_x, plotsize_y * numplots_y)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    return fig, axes
# Getting the data.

train_file_name = "../input/liverpool-ion-switching/train.csv"

train_data = np.loadtxt(train_file_name, dtype=np.float32, skiprows=1, delimiter=',', usecols=(1,2))

train_answers = train_data[:,1].astype(dtype=np.int32)

train_data = train_data[:,0]



train_batches = {0: (0, 5*10**5), 1: (5*10**5, 6*10**5), 2: (6*10**5,10**6)}

train_batches.update({i: (5*10**5*(i-1), 5*10**5*i) for i in range(3, 8)})

train_batches.update({8: (35*10**5, 3642000), 9: (3642000, 3824000), 10: (3824000, 4*10**6)})

train_batches.update({i: (5*10**5*(i-3), 5*10**5*(i-2)) for i in range(11, 13)})

drift_train_batches = (1, 7, 11, 12) # manually do 8,9,10

diff_limits = {'1r': 1.43, '1f': 1.43, '3': 1.43, '5': 1.43, '1-5': 1.43, '10': 2.01}

models_train_precise = {'1r':(0, 1, 2), '1f': (3, 7), '3': (4, 8, 10), '5': (6, 11), '10': (5, 12)}



def cut_shape(n_cuts, cut_len):

    return [(10**4*cut_len*i, 10**4*cut_len*(i + 1)) for i in range (n_cuts)]

cuts_train = {0:cut_shape(5,10), 1:cut_shape(1,10), 2:cut_shape(4,10), # 1s

              3:cut_shape(50,1), 7:cut_shape(50,1), #1f

              4:cut_shape(5,10), 8:[(0,10**5), (10**5,142000)], 9:[(0,58000),(58000,158000),(158000,182000)],

              10:[(0,76000), (76000,176000)], # 3

              6:cut_shape(5,10), 11:cut_shape(5,10), # 5

              5:cut_shape(5,10), 12:cut_shape(5,10)} #10



# Remove drift. Credits to Chris Deotte

drift_arr = np.sin(np.arange(0,np.pi+1.e-6,np.pi/499999))*5.

clean_train_data = np.copy(train_data)

for i in drift_train_batches:

    i_start, i_end = train_batches[i]

    clean_train_data[i_start: i_end] -= drift_arr[:i_end-i_start]

clean_train_data[35*10**5: 40*10**5] -= drift_arr # manualy on batch 8-10

float_train_answers = train_answers.astype(np.float32)
lin_reg=dict()

noise_period=200



def make_diff_features(signal):

    signal0 = deepcopy(signal)

    signal1 = np.diff(signal0, prepend=[signal0[0]]) * -1.

    signal2 = signal1 - np.diff(signal0[:-1], prepend=[signal0[0]]*2)

    return np.stack([signal0, signal1, signal2], axis=-1)



def prepare_train_signal(key, to_cut=(0,0), signal=None):

    lin_fit_train_answers = []

    lin_fit_train_signal = []

    if signal is None:

        signal = clean_train_data

    for i in models_train_precise[k]:

        i_start, i_end = train_batches[i]

        for di_start, di_end in cuts_train[i]:

            lin_fit_train_answers.append(make_diff_features(

                float_train_answers[i_start+di_start:i_start+di_end-to_cut[1]])[to_cut[0]:])

            lin_fit_train_signal.append(signal[i_start+di_start+to_cut[0]:i_start+di_end-to_cut[1]])

    return np.concatenate(lin_fit_train_answers), np.concatenate(lin_fit_train_signal)



def plot_noise_fourier(noise_signal, cuts_seq, half_window_size):

    """cuts is a sequence of sequences, each of the subsequences starts from 0 (0,i1),(i1,i2),..."""

    coss = np.cos(np.arange(half_window_size * 2) * 2. * math.pi / noise_period)

    sins = np.sin(np.arange(half_window_size * 2) * 2. * math.pi / noise_period)

    fig, (ax1, ax2) = create_axes_grid(2,1,20,6)

    n_c = 0

    i_s = 0

    for cuts in cuts_seq:

        for c1,c2 in cuts:

            xs = []

            As = []

            phis = []

            k_phi = 0

            for i_med in range(i_s + half_window_size, i_s + c2 - c1, half_window_size):

                i_start = i_med - half_window_size

                i_end = min(i_med + half_window_size, i_s + c2 - c1)

                if i_end < i_start + noise_period:

                    break

                xs.append((i_start + i_end)/20000.)

                a = np.sum(noise_signal[i_start:i_end] * coss[:i_end-i_start])/(i_end-i_start)*2.

                b = np.sum(noise_signal[i_start:i_end] * sins[:i_end-i_start])/(i_end-i_start)*2.

                As.append(np.sqrt(a*a + b*b))

                phi = np.arctan(max(np.abs(b), 1.e-5)*np.sign(b) / max(np.abs(a), 1.e-5)*np.sign(a))/math.pi

                if len(phis) == 0:

                    pass

                elif phi + k_phi - phis[-1] > 0.5:

                    k_phi -= 1

                elif phis[-1] - phi - k_phi > 0.5:

                    k_phi += 1

                if len(phis) > 1 and phis[-1] - phis[-2] > 0.35 and phi + k_phi - phis[-1] > 0.35:

                    k_phi -= 1

                if len(phis) > 1 and phis[-2] - phis[-1] > 0.35 and phis[-1] - phi - k_phi > 0.35:

                    k_phi += 1

                phis.append(phi + k_phi)

            ax1.plot(xs, As, c=('b' if n_c % 2 == 0 else 'r'))

            ax1.set_title('50Hz harmonic magnitude on t')

            for delta in np.arange(-5, 6):

                ax2.plot(xs, phis + delta, c=('b' if n_c % 2 == 0 else 'r'))

            n_c += 1

            i_s += c2-c1

    ax2.set_ylim((-1.5,1.5))

    ax1.grid(True)

    ax2.grid(True)

    ax2.set_title('50 Hz harmonic phase on t')

    plt.show()
for k in models_train_precise.keys():

    #Part 1: Initial predictions

    lin_fit_train_answers, lin_fit_train_signal = prepare_train_signal(k, to_cut=(2,0))

    lin_reg[k] = LinearRegression()

    for i in range(3): # 3 iterations of improval

        lin_reg[k].fit(lin_fit_train_answers, lin_fit_train_signal)

        predictions = lin_reg[k].predict(lin_fit_train_answers)

        difference = predictions - lin_fit_train_signal

        lin_fit_train_signal += difference * (np.abs(difference) > diff_limits[k])

    lin_fit_train_answers, lin_fit_train_signal = prepare_train_signal(k)

    predicted_train_signal = lin_reg[k].predict(lin_fit_train_answers)

    train_noise = lin_fit_train_signal - predicted_train_signal

    predicted_signal_var = np.var(predicted_train_signal)

    train_noise_var = np.var(train_noise)

    print("Model {}: intercept={:.4f}, slope={}, signal_var={:.4f}, noise_var={:.4f}".

          format(k, lin_reg[k].intercept_, lin_reg[k].coef_, predicted_signal_var, train_noise_var))    

    train_half_window_size = (2000 if k == '10' else 800)

    plot_noise_fourier(train_noise, [cuts_train[i] for i in models_train_precise[k]], train_half_window_size)
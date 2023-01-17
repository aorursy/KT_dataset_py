import numpy as np

import librosa.display

import tensorflow as tf

import tensorflow.contrib.eager as tfe



tfe.enable_eager_execution()



from scipy.io import wavfile



import matplotlib.pyplot as plt

%matplotlib inline
def hz_to_mel(freq):

  return 1127. * tf.log(1.0 + (freq / 700.))



def mel_to_hz(mel):

  return 700.*(tf.exp(mel/1127.)-1.)



def multi_ffts_to_mel(freq_array, n_mels=128):

  melfreq_array = tf.expand_dims(hz_to_mel(freq_array),0)

  

  mel_edges = tf.lin_space(hz_to_mel(tf.reduce_min(freq_array)), #or just use 0

                           hz_to_mel(tf.reduce_max(freq_array)), #or SR/2

                           n_mels+2)

  

  lower_edge_mel, center_mel, upper_edge_mel =tf.split(tf.contrib.signal.frame(mel_edges, 3, 1, axis=-1), 3, axis=-1)



  wt_down = (melfreq_array - lower_edge_mel) / (center_mel - lower_edge_mel)

  wt_up = (upper_edge_mel - melfreq_array) / (upper_edge_mel - center_mel)

  

  mel_weights_matrix = tf.maximum(0.0, tf.minimum(wt_down, wt_up))

  center_mel_freqs = mel_to_hz(center_mel) 

  

  return mel_weights_matrix, center_mel_freqs



def audioframes2logmelspec(b_framed_signal, n_ffts=5, 

                           wvls_per_window_hinge=16, n_mel=128, 

                           fft_l1=1024, sr=16000):

  # batch_framed_signal has shape: (batch_size x n_windows x fft_l1)

  # decrease weights for samples w/ more than wvls_per_window_hinge

  # wvls_per_window_hinge method could be improved, maybe weight~pmf of poisson?

    

  fft1_space = tf.lin_space(0., .5, 1+fft_l1//2)[1:]

  freq_list =[sr*fft1_space] 

  n_wv_list =[fft_l1*fft1_space]



  fft_list =[tf.spectral.rfft(b_framed_signal)[:,:,1:]]

  

  for i in range(1,n_ffts):

    fft_lnew = fft_l1//2**i

    fftnew_space = tf.lin_space(0., .5, 1+fft_lnew//2)[1:]

    

    freq_list.append(sr*fftnew_space)

    n_wv_list.append(fft_lnew*fftnew_space)

    

    frames_new = b_framed_signal[:, :, (fft_l1-fft_lnew)//2:(fft_l1-fft_lnew)//2+fft_lnew]

    fft_list.append(tf.spectral.rfft(frames_new)[:,:,1:])

    

  

  freq_concat = tf.concat(freq_list, axis=-1)

  n_wv_concat = tf.concat(n_wv_list, axis=-1)

  fft_concat = tf.concat(fft_list, axis=-1)

    

  magnitude_spectros = tf.abs(fft_concat)



  mel_wts, center_mel_freqs = multi_ffts_to_mel(freq_concat, n_mel)

  wvls_wts = tf.where(n_wv_concat>wvls_per_window_hinge, wvls_per_window_hinge/n_wv_concat, tf.ones_like(n_wv_concat))

  

  mel_spectro=tf.tensordot(magnitude_spectros, (mel_wts*tf.expand_dims(wvls_wts,0)),axes = [[2], [1]])



  log_mel_spectro = tf.log(mel_spectro+1e-7)

  

  return tf.expand_dims(log_mel_spectro, -1), center_mel_freqs

some_paths = [

'./data/train/audio/marvin/8625475c_nohash_0.wav',

'./data/train/audio/tree/8625475c_nohash_1.wav',  

'./data/train/audio/tree/8625475c_nohash_2.wav',   

'./data/train/audio/tree/8625475c_nohash_3.wav',

'./data/train/audio/no/8625475c_nohash_0.wav', 

'./data/train/audio/zero/8625475c_nohash_0.wav',

'./data/train/audio/zero/8625475c_nohash_1.wav',

'./data/train/audio/down/8625475c_nohash_0.wav']
def plot_several_logmelspec(paths):

  n=len(paths)



  plt.figure(figsize=(12,4*n))



  for i, path in enumerate(paths):

    plt.subplot(n, 1, i+1)



    sr, wav = wavfile.read(path)

    signal = wav.astype(np.float32) / np.iinfo(np.int16).max



    b_signals = tf.expand_dims(signal, axis=0)



    b_framed_signal = tf.contrib.signal.frame(b_signals, 

                                          frame_length=1024, 

                                          frame_step = 32)

    log_mel_spectro, center_mel_freqs = audioframes2logmelspec(b_framed_signal, sr=sr)



    librosa.display.specshow(log_mel_spectro[0,:,:,0].numpy().T, sr=sr, x_axis='time', 

                             y_axis='mel', hop_length=32, 

                             fmin=tf.reduce_min(center_mel_freqs), 

                             fmax=tf.reduce_max(center_mel_freqs), 

                             cmap='coolwarm')



    plt.title(path)

    plt.colorbar(format='%+02.0f dB')



  plt.tight_layout()
plot_several_logmelspec(some_paths)
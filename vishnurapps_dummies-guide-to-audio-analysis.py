import librosa

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/purfin/XC195200.mp3", offset=30, duration=5)

print(librosa.feature.mfcc(y=y, sr=sr))
plt.figure(figsize=(20, 4))

plt.plot(librosa.feature.mfcc(y=y, sr=sr))

plt.grid()

plt.title('Mel-frequency cepstral coefficients')

plt.xlabel('time')

plt.ylabel('mfcc coefficients')

plt.show()
y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/purfin/XC195200.mp3")

print(librosa.feature.zero_crossing_rate(y))
plt.figure(figsize=(20, 4))

plt.plot(librosa.feature.zero_crossing_rate(y).squeeze())

plt.grid()

plt.title('Zero crossing rate')

plt.xlabel('time')

plt.ylabel('the fraction of zero crossings in the i th frame')

plt.show()
S, phase = librosa.magphase(librosa.stft(y))

print(librosa.feature.spectral_rolloff(S=S, sr=sr))
plt.figure(figsize=(20, 4))

plt.plot(librosa.feature.spectral_rolloff(S=S, sr=sr).squeeze())

plt.grid()

plt.title('Roll off frequency')

plt.xlabel('time')

plt.ylabel('Hz')

plt.show()
import librosa

y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/purfin/XC195200.mp3", duration=10.0)

onset_env = librosa.onset.onset_strength(y=y, sr=sr)

print(onset_env)
plt.figure(figsize=(20, 4))

plt.plot(onset_env)

plt.grid()

plt.title('Spectral flux')

plt.xlabel('time')

plt.ylabel('Onset')

plt.show()
y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/purfin/XC195200.mp3")

print(librosa.feature.chroma_stft(y=y, sr=sr))
plt.figure(figsize=(20, 4))

plt.plot(librosa.feature.chroma_stft(y=y, sr=sr).squeeze())

plt.grid()

plt.title('Chroma feature')

plt.xlabel('time')

plt.ylabel('energy for each chroma bin at each frame')

plt.show()
y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/purfin/XC195200.mp3")

pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

print(pitches)
plt.figure(figsize=(20, 4))

plt.plot(pitches.squeeze())

plt.grid()

plt.title('Pitch')

plt.show()
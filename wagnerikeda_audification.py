# listen to recordings of NBIRDS at the same time

import numpy as np
from pathlib import Path
from pydub import AudioSegment
from IPython.display import FileLink

PATH_TRAIN = "/kaggle/input/birdsong-recognition/train_audio"
LENGTH = 60 #seconds
SAMPLE_RATE = 2**15

def get_birds():
    path = Path(PATH_TRAIN)
    birds = [str(f).split("/")[-1] 
             for f in path.rglob("*") 
             if f.is_dir()]
    birds.sort()
    return birds

def get_audio(bird, length=LENGTH):
    path = Path(PATH_TRAIN)/bird
    files = [f for f in path.rglob("*.mp3")]
    np.random.shuffle(files)
    all_audio = AudioSegment.empty()
    while len(all_audio)/1e3<length:
        f = files.pop()
        audio = AudioSegment.from_file(f)
        all_audio += audio
    all_audio = all_audio[:length*1e3]
    return all_audio

def overlay_birds(birds, length=LENGTH):
    pans = [2*i/(len(birds)-1)-1 for i in range(len(birds))]
    combined = AudioSegment.silent(length*1e3)
    for b in birds:
        sound_bird = get_audio(b, length)
        #sound_bird.pan(pans.pop())
        combined = combined.overlay(sound_bird)
    return combined
birds = get_birds()
NBIRDS = 10
listen_to = list(np.random.choice(birds, NBIRDS, replace=False))
print(listen_to)
audification = overlay_birds(listen_to, length=30)
audification
filename = "_".join(listen_to) + ".wav"
audification.export(filename)
FileLink(filename)

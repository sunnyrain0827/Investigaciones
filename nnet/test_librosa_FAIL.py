import librosa
import numpy as np
import os

def audio_from_file(x, dur=10):
   flen = os.stat(x).st_size /2
   print("flen: " + str(flen))
   ix = np.random.randint(0, flen - 40000)/8000
   print(ix)
   y, sr = librosa.load(x, sr=8000, offset=ix, duration=5.0)
   features = librosa.feature.mfcc(y=y,sr=sr)
   return features




if __name__ == '__main__':

    path = "/Users/iorife/github/kaldi-iorife/data/audio/tmobile_01.wav"
    features = audio_from_file(path)
    print(features.shape)
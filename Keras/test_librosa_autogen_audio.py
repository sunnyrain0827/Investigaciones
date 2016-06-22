# We'll need numpy for some mathematical operations
import numpy as np
import librosa

import matplotlib.pyplot as plt

def createFourOnTheFloor(bpm, nb_beats, sr, sample):

    # 60s/min * min/beats(1/bpm) * sr * nb_beats == seconds/beat * samples/second * nb_beats
    # == samples/beat * nb_beats

    samples_per_beat = 60.0 / bpm * sr
    total_loop_len = int(samples_per_beat * nb_beats)

    mixing_bed = np.zeros(int(total_loop_len))
    print("Size of mixing bed: " + str(len(mixing_bed)))

    offset = 0
    i = 0
    while (offset < total_loop_len):

        print("offset: " + str(offset) + " i: " + str(i) + " " + str(float(offset)/sr))
        if(offset + len(sample) > total_loop_len):
            mixing_bed[offset:total_loop_len - 1] += sample[0:total_loop_len - offset - 1]
            break

        mixing_bed[offset:offset + len(sample)] += sample
        offset += int(samples_per_beat)
        i += 1

    return mixing_bed


x = "/Users/iorife/github/Investigaciones/Drums/Kick/Kick3kDeep.wav"
sample, sr = librosa.load(x, mono=True, sr=48000)
print("Len of samples: " + str(len(sample)))
y = createFourOnTheFloor(bpm=128, nb_beats=32, sr=sr, sample=sample)

librosa.output.write_wav("4-on-floor-output.wav", y, sr)

# Run the beat tracker.
#def beat_track(y=None, sr=22050, onset_envelope=None, hop_length=512,
#               start_bpm=120.0, tightness=100, trim=True, bpm=None):

tempo, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=128, hop_length=512)
print("\nBeat times: ", librosa.frames_to_time(beats, sr=sr))
print("Num beats: " + str(len(beats)))
print("Tempo: " + str(tempo))

# plt.figure(figsize=(12, 6))
# plt.interactive(False)
# plt.plot(y)
# plt.show(block=True)
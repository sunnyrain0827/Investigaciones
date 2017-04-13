import numpy as np

def createFourOnTheFloor(bpm, nb_beats, sr, sample):

    # 60s/min * min/beats(1/bpm) * sr * nb_beats == seconds/beat * samples/second * nb_beats
    # == samples/beat * nb_beats

    samples_per_beat = int(60.0 / bpm * sr)
    total_loop_len = int(samples_per_beat * nb_beats)
    mixing_bed = np.zeros(int(total_loop_len))

    # we assume samples are one shots, triggered on each beat
    # trim samples to beat length
    sample = sample[0:samples_per_beat]
    trimmed_sample_len = len(sample)

    # print("samples per beat: " + str(samples_per_beat))
    # print("[createFourOnTheFloor] Size of mixing bed: " + str(len(mixing_bed)))
    # print("len sample : " + str(len(sample)))
    # print("trimmed sample len : " + str(trimmed_sample_len))

    # apply a 1000 samples (21ms) fade-out to minimize pops
    len_fade_out = 240
    fade_out_vector = np.arange(0, len_fade_out, 1)
    fade_out = np.exp(-fade_out_vector/160.)

    # print(len(sample[(trimmed_sample_len - len_fade_out):]))
    sample[(trimmed_sample_len - len_fade_out):] = sample[(trimmed_sample_len - len_fade_out):] * fade_out

    offset = 0
    i = 0
    while (offset < total_loop_len):
        # print("offset: " + str(offset) + " i: " + str(i) + " " + str(float(offset)/sr))
        mixing_bed[offset:offset + len(sample)] += sample
        offset += int(samples_per_beat)
        i += 1

    return mixing_bed

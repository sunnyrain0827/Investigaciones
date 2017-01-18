import numpy as np

def createFourOnTheFloor(bpm, nb_beats, sr, sample):

    # 60s/min * min/beats(1/bpm) * sr * nb_beats == seconds/beat * samples/second * nb_beats
    # == samples/beat * nb_beats

    samples_per_beat = 60.0 / bpm * sr
    total_loop_len = int(samples_per_beat * nb_beats)

    mixing_bed = np.zeros(int(total_loop_len))
    # print("[createFourOnTheFloor] Size of mixing bed: " + str(len(mixing_bed)))

    offset = 0
    i = 0
    while (offset < total_loop_len):

        # print("offset: " + str(offset) + " i: " + str(i) + " " + str(float(offset)/sr))
        if(offset + len(sample) > total_loop_len):
            mixing_bed[offset:total_loop_len - 1] += sample[0:total_loop_len - offset - 1]
            break

        mixing_bed[offset:offset + len(sample)] += sample
        offset += int(samples_per_beat)
        i += 1

    return mixing_bed
# Wikipedia implementation https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
#


def fwd_bkw(x, states, a_0, a, e, end_st):
    L = len(x)

    fwd = []
    f_prev = {}
    # forward part of the algorithm
    for i, x_i in enumerate(x):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = a_0[st]
            else:
                prev_f_sum = sum(f_prev[k] * a[k][st] for k in states)

            f_curr[st] = e[st][x_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * a[k][end_st] for k in states)

    bkw = []
    b_prev = {}
    # backward part of the algorithm
    for i, x_i_plus in enumerate(reversed(x[1:] + (None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = a[st][end_st]
            else:
                b_curr[st] = sum(a[st][l] * e[l][x_i_plus] * b_prev[l] for l in states)

        bkw.insert(0, b_curr)
        b_prev = b_curr

    p_bkw = sum(a_0[l] * e[l][x[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(L):
        posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    print("p_fwd: " + str(p_fwd) + "  p_bkw: " + str(p_bkw))
    assert p_fwd == p_bkw
    return fwd, bkw, posterior


## FWD-BKWD
states = ('Healthy', 'Fever')
end_state = 'E'

# observations = ('normal', 'cold', 'dizzy')
observations = ('normal', 'cold', 'dizzy', 'cold', 'cold', 'cold')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
    'Healthy': {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
    'Fever': {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
}
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


def example():
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)


# [IOHAVOC] run fwd_bkwd
# for line in example():
#     print(*line)


#############################

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for i in states:
        V[0][i] = start_p[i] * emit_p[i][obs[0]]
        print("V[0][" + i + "]=" + str(V[0][i]))

    print("\n")
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for y in states:
            prob = max(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]] for y0 in states)
            V[t][y] = prob

    for i in dptable(V):
        print(i)
    opt = []
    for j in V:
        for x, y in j.items():
            if j[x] == max(j.values()):
                opt.append(x)

    # The highest probability
    h = max(V[-1].values())
    print('\nThe steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % h)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%10d" % i) for i in range(len(V)))
    for y in V[0]:
        yield "%.7s: " % y + " ".join("%.7s" % ("%f" % v[y]) for v in V)


#
viterbi(observations, states,
        start_probability,
        transition_probability,
        emission_probability)

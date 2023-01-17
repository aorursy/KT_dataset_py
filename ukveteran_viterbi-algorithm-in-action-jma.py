def viterbi(sequence_of_observations, states, initial_probabilities, transition_probabilities, emission_probabilities):

    V = [{}]

    for s in states:

        V[0][s] = {

            'probability': initial_probabilities[s] * emission_probabilities[s][sequence_of_observations[0]],

            'previous': None}

    for t in range(1, len(sequence_of_observations)):

        V.append({})

        for s in states:

            max_tr_probability = V[t-1][states[0]]['probability'] * transition_probabilities[states[0]][s]

            previous_s_selected = states[0]

            for previous_s in states[1:]:

                tr_probability = V[t-1][previous_s]['probability'] * transition_probabilities[previous_s][s]

                if tr_probability > max_tr_probability:

                    max_tr_probability = tr_probability

                    previous_s_selected = previous_s        

            max_probability = max_tr_probability * emission_probabilities[s][sequence_of_observations[t]]

            V[t][s] = {'probability': max_probability, 'previous': previous_s_selected}                    

    most_likely_states = []

    max_probability = max(value['probability'] for value in V[-1].values())

    previous = None

    for s, data in V[-1].items():

        if data['probability'] == max_probability:

            most_likely_states.append(s)

            previous = s

            break

    for t in range(len(V) - 2, -1, -1):

        most_likely_states.insert(0, V[t + 1][previous]['previous'])

        previous = V[t + 1][previous]['previous']

    return {'steps': most_likely_states, 'max_probability': max_probability}
sequence_of_observations = ['Heads', 'Tails', 'Tails', 'Heads', 'Tails', 'Heads', 'Heads', 'Heads', 'Tails', 'Heads']

states = ['Fair', 'Loaded']

initial_probabilities = {'Fair': 0.6, 'Loaded': 0.4}

transition_probabilities = {

    'Fair': {'Fair': 0.6, 'Loaded': 0.4},

    'Loaded': {'Fair': 0.4, 'Loaded': 0.6}

}

emission_probabilities = {

    'Fair': {'Heads': 0.5, 'Tails': 0.5},

    'Loaded': {'Heads': 0.8, 'Tails': 0.2}

}
viterbi(sequence_of_observations,

        states,

        initial_probabilities,

        transition_probabilities,

        emission_probabilities)
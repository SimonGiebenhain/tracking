import numpy as np

class MarkovChain():
    def __init__(self, states, transition_list):
        assert len(states) == len(transition_list)
        P = np.zeros([len(states), len(states)])
        for i, p_list in enumerate(transition_list):
            assert len(p_list) == len(states) - 1
            P[i, i] = 1 - sum(p_list)
            for j, p in enumerate(p_list):
                if j >= i:
                    P[i, j+1] = p_list[j]
                else:
                    P[i, j] = p_list[j]

        self.states = states
        self.P = P
        self.state = None

    def set_state(self, state):
        self.state = np.reshape(state, [1, np.shape(self.P)[0]])

    def take_step(self):
        self.state = self.state @ self.P
        return self.state

    def take_n_steps(self, n):
        self.state = self.state @ np.linalg.matrix_power(self.P, n)
        return self.state

    def sample_rollouts(self, initial_state, n_rollouts, T):
        state_history = np.zeros([n_rollouts, T])
        self.set_state(initial_state)
        for t in range(T):
            s = np.squeeze(self.take_step())
            state_history[:, t] = np.argmax(np.random.multinomial(1, s, n_rollouts), axis=1)
            #print(np.random.choice(np.arange(1, np.shape(self.states)[0]), [n_rollouts], True, self.state.T))
        return state_history



noise_model_states = ['not flying, no wings', 'not flying, wing covers', 'flying, no wings' ,'flying, wing covers']

p_nF_nFW = 0.004
p_nF_F = 0.002
p_nF_FW = 0.002

p_nFW_nF = 0.05
p_nFW_F = 0.002
p_nFW_FW = 0.002

p_F_nF = 0.01
p_F_nFW = 0.01
p_F_FW = 0.03

p_FW_nF = 0.01
p_FW_nFW = 0.01
p_FW_F = 0.03

noise_model_transition_prob = [[p_nF_nFW, p_nF_F, p_nF_FW],
                               [p_nFW_nF, p_nFW_F, p_nFW_FW],
                               [p_F_nF, p_F_nFW, p_F_FW],
                               [p_FW_nF, p_FW_nFW, p_FW_F]]

behaviour_model_states = ['sitting', 'starting', 'flying', 'landing', 'walking']
p_sit_start = 0.0005
p_sit_walk = 0.002

p_start_fly = 0.1

p_fly_land = 0.005

p_land_sit = 0.05
p_land_walk = 0.05

p_walk_sit = 0.01
p_walk_start = 0.0005

behaviour_model_transition_prob = [[p_sit_start, 0, 0, p_sit_walk],
                                   [0, p_start_fly, 0, 0],
                                   [0, 0, p_fly_land, 0],
                                   [p_land_sit, 0, 0, p_land_walk],
                                   [p_walk_sit, p_walk_start, 0, 0]]

mc_behaviour_model = MarkovChain(behaviour_model_states, behaviour_model_transition_prob)
mc_behaviour_model.set_state(np.array([1, 0, 0, 0, 0]))
print(mc_behaviour_model.P)
print(mc_behaviour_model.sample_rollouts(np.array([1, 0, 0, 0, 0]), 2, 100))
mc_noise_model = MarkovChain(noise_model_states, noise_model_transition_prob)
#print((mc_noise_model.P))

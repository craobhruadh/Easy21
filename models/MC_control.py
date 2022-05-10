from environment.easy21 import Easy21
from collections import Counter
import numpy as np


class MonteCarloControl:

    def __init__(self, discount=1, N0=100):
        self.discount = discount
        self.N0 = N0
        self.state_visit_counter = Counter()
        self.state_action_counter = Counter()
        self.possible_actions = ['stick', 'hit']
        self.value = {
            'stick': np.zeros((21, 10)),
            'hit': np.zeros((21, 10))
        }
        self.returns = []
        self.game = Easy21()

    def get_alpha(self, state, action):
        return 1.0/self.state_visit_counter[(action, state)]

    def get_epsilon(self, state):
        return self.N0/(self.N0+self.state_visit_counter[state])

    def train(self, n_episodes=1000):
        """
        Use a "greedy policy" when not in exploitation mode: choose
        the state that maximizes expected reward
        """
        for i in range(n_episodes):
            episodes = self.model_episode()
            self.update_value_function(episodes)

    def update_value_function(self, episodes):
        G = 0
        returns = []
        for i in range(len(episodes)-1, -1, -1):
            state, action, reward = episodes[i]
            G = G * self.discount + reward
            returns.append(G)
            alpha = self.get_alpha(state, action)
            self.value[action][state[0]-1][state[1]-1] += (
                alpha*(np.mean(returns)-self.value[action][state[0]-1][state[1]-1])
            )

    def model_episode(self):
        """
        Returns an array of state/action/reward pairs
        """
        self.game.setup_new_game()
        episodes = []
        while not self.game.terminal:
            state = self.game.get_state()
            self.state_visit_counter[state] += 1
            if np.random.rand() < self.get_epsilon(state):
                action = np.random.choice(self.possible_actions)
            elif (
                self.value['stick'][state[0]-1][state[1]-1]
                > self.value['hit'][state[0]-1][state[1]-1]
            ):
                action = 'stick'
            else:
                action = 'hit'

            reward = self.game.step(action)
            self.state_visit_counter[(action, state)] += 1
            episodes.append((state, action, reward))
        return episodes

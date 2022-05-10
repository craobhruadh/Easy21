from environment.easy21 import Easy21
from collections import Counter
import numpy as np


class TDLearning:

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

    def get_lambda(self):
        return np.arange(0, 1, 0.1)

    def train(self, n_episodes=1000):
        """
        Use a "greedy policy" when not in exploitation mode: choose
        the state that maximizes expected reward
        """
        for i in range(n_episodes):
            self.model_episode()

    def update_value_function(self, state, state_prime, action, action_prime, reward):
        """Updates the value function as expected for SARSA.  Note that
        it is designed to be updated after each action, not after each
        episode (like in MC Control)
        """
        Q = self.value[action][state[0]-1][state[1]-1]
        alpha = self.get_alpha(state, action)
        Q_prime = self.value[action_prime][state_prime[0]-1][state_prime[1]-1]
        error = alpha * (reward + self.discount*Q_prime - Q)
        self.value[action][state[0]-1][state[1]-1] = Q + error

    def get_next_action(self, state):
        """
        This is effectively our policy.  If we are in explore mode, choose
        a random action from our policy.  If in exploitation mode, choose
        the action that is greedily expected to maximize our return.
        """
        if np.random.rand() < self.get_epsilon(state):
            action = np.random.choice(self.possible_actions)
        elif (
            self.value['stick'][state[0]-1][state[1]-1]
            > self.value['hit'][state[0]-1][state[1]-1]
        ):
            action = 'stick'
        else:
            action = 'hit'
        return action

    def model_episode(self):
        """
        Returns an array of state/action/reward pairs
        """
        self.game.setup_new_game()
        state = self.game.get_state()
        self.state_visit_counter[state] += 1
        action = self.get_next_action(state)
        while not self.game.terminal:
            reward = self.game.step(action)
            state_prime = self.game.get_state()
            self.state_visit_counter[state_prime] += 1
            self.state_visit_counter[(action, state)] += 1
            if self.game.terminal:
                alpha = self.get_alpha(state, action)
                self.value[action][state[0]-1][state[1]-1] += alpha * reward
            else:
                action_prime = self.get_next_action(state_prime)
                self.update_value_function(
                    state,
                    state_prime,
                    action,
                    action_prime,
                    reward
                )
                state = state_prime
                action = action_prime

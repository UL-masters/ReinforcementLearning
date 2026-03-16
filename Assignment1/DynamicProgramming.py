#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Assignment1.Environment import StochasticWindyGridworld
from Assignment1.Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s '''
        return argmax(self.Q_sa[s])
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        q_old = self.Q_sa[s, a]
        q_new = 0.0

        for s_next in range(self.n_states):
            if p_sas[s_next] > 0:
                q_new += p_sas[s_next] * (
                        r_sas[s_next] + self.gamma * np.max(self.Q_sa[s_next])
                )

        self.Q_sa[s, a] = q_new

        return abs(q_old - q_new)
    
    
def Q_value_iteration(env, gamma=1.0, threshold=0.001):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma)

    i = 0
    while True:
        max_error = 0

        for s in range(env.n_states):
            for a in range(env.n_actions):
                p_sas, r_sas = env.model(s, a)
                error = QIagent.update(s, a, p_sas, r_sas)
                max_error = max(max_error, error)

        env.render(Q_sa=QIagent.Q_sa,
                   plot_optimal_policy=True,
                   step_pause=0.2)

        print("Q-value iteration, iteration {}, max error {}".format(i, max_error))

        i += 1

        if max_error < threshold:
            break
 
    return QIagent

def experiment():
    gamma = 1.0
    threshold = 0.001
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold)

    done = False
    s = env.reset()
    while not done:
        a = QIagent.select_action(s)
        s_next, r, done = env.step(a)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next

    start_state = 3
    V_start = np.max(QIagent.Q_sa[start_state])

    goal_reward = 100
    step_reward = -1.0

    N = (V_start - goal_reward) / step_reward + 1

    mean_reward_per_timestep = V_start / N

    print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    
if __name__ == '__main__':
    experiment()

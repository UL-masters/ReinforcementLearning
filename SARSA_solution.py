#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class SarsaAgent(BaseAgent):
        
    def update(self,s,a,r,s_next,a_next,done):
        # current state, action, reward, next state, next action, done flag
        
        # current Q value
        current_q = self.Q_sa[s][a]

        # compute target (Gt)
        # reward plus the discounted value of the next action
        if done:
            target = r
        else:
            target = r + self.gamma * self.Q_sa[s_next][a_next]

        # SARSA update
        # move the current Q-value a bit towards the target
        self.Q_sa[s][a] = current_q + self.learning_rate * (target - current_q)

        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    # TO DO: Write your SARSA algorithm here!

    # initialize state and action
    s = env.reset()
    if policy == 'egreedy':
        a = pi.select_action(s, policy='egreedy', epsilon=epsilon)
    elif policy == 'softmax':
        a = pi.select_action(s, policy='softmax', temp=temp)
    
    # loop over budget = timesteps
    for t in range(1, n_timesteps + 1):

        # simulate environment
        s_next, r, done = env.step(a)

        # sample action (on-policy)
        if not done:
            if policy == 'egreedy':
                a_next = pi.select_action(s_next, policy='egreedy', epsilon=epsilon)
            elif policy == 'softmax':
                a_next = pi.select_action(s_next, policy='softmax', temp=temp)
        else:
            a_next = None

        # update Q-values
        pi.update(s, a, r, s_next, a_next, done)

        # move to next state, reset environment
        if done:
            s = env.reset()
            if policy == 'egreedy':
                a = pi.select_action(s, policy='egreedy', epsilon=epsilon)
            elif policy == 'softmax':
                a = pi.select_action(s, policy='softmax', temp=temp)
        else:
            s = s_next
            a = a_next

        # evaluation
        if t % eval_interval == 0:
            mean_return = pi.evaluate(eval_env)
            eval_returns.append(mean_return)
            eval_timesteps.append(t)


    # if plot:
    #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return np.array(eval_returns), np.array(eval_timesteps) 


def test():
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True
    sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
            
    
if __name__ == '__main__':
    test()

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

class MonteCarloAgent(BaseAgent):
        
    def update(self, states, actions, rewards):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        G = 0.0  # this is G_{t+1} = 0
        
        # loop backwards
        for t in reversed(range(len(actions))):
            
            G = rewards[t] + self.gamma * G
            
            s = states[t]
            a = actions[t]
            
            # Monte Carlo update
            self.Q_sa[s, a] += self.learning_rate * (G - self.Q_sa[s, a])

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    timestep = 0

    while timestep < n_timesteps: # budget = timesteps
        
        # sample initial state
        s = env.reset()
        states = [s]
        actions = []
        rewards = []
        
        done = False
        episode_length = 0
        
        # collect episode
        while not done and episode_length < max_episode_length and timestep < n_timesteps:
            
            a = pi.select_action(s, policy, epsilon, temp)
            s_next, r, done = env.step(a)
            
            states.append(s_next)
            actions.append(a)
            rewards.append(r)
            
            s = s_next
            
            timestep += 1
            episode_length += 1
            
            # evaluation
            if timestep % eval_interval == 0:
                mean_return = pi.evaluate(eval_env)
                eval_timesteps.append(timestep)
                eval_returns.append(mean_return)
        
        # after episode → update
        pi.update(states, actions, rewards)

        
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

                 
    return np.array(eval_returns), np.array(eval_timesteps) 
    
def test():
    n_timesteps = 1000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    
            
if __name__ == '__main__':
    test()

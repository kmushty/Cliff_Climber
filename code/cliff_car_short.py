# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:03:56 2020

@author: Sri Sai Kaushik
"""
# objective is to get the cart to the flag.
# for now, let's just move randomly:

from my_mountain_car_short import MountainCarEnv
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

env = MountainCarEnv()

# parameter definition

l_rate = 0.1
discount_factor = 0.95
output_of = 600
begin_eps_decay = 1
plot_of = 100
Obs_space = [40] * len(env.observation_space.high)
Obs_space_win = (env.observation_space.high - env.observation_space.low)/Obs_space
episodes = 8000
epsilon = 0.9 
end_eps_decay = episodes//2
epsilon_decay_value = epsilon/(end_eps_decay - begin_eps_decay)

#define the q_table 

q_table = np.random.uniform(low=-2, high=0, size=(Obs_space + [env.action_space.n]))

# plot def

epsiode_rew = []
cumu_epsiode_rew = {'max': [], 'min': [], 'ep': [], 'avg': []}


def get_state(state):
    act_state = (state - env.observation_space.low)/Obs_space_win
    final_act_state = act_state.astype(np.int)
    return tuple(final_act_state)


for ep in range(episodes):
    episode_reward = 0
    act_state = get_state(env.reset())
    done = False

    if ep % output_of == 0:
        render = True
        print(ep)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[act_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_act_state = get_state(new_state)

        if ep % output_of == 0:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_act_state])
            current_q = q_table[act_state + (action,)]
            update_q = (1 - l_rate) * current_q + l_rate * (reward + discount_factor * max_future_q)
            
            q_table[act_state + (action,)] = update_q
        elif new_state[0] >= env.goal_position:
            q_table[act_state + (action,)] = 0
            
        act_state = new_act_state
        
    if begin_eps_decay <= ep <= end_eps_decay:
        epsilon -= epsilon_decay_value
    epsiode_rew.append(episode_reward)    
    
        
    if not ep % plot_of:
        average_reward = sum(epsiode_rew[-plot_of:])/plot_of
        cumu_epsiode_rew['ep'].append(ep)
        cumu_epsiode_rew['avg'].append(average_reward)
        cumu_epsiode_rew['max'].append(max(epsiode_rew[-plot_of:]))
        cumu_epsiode_rew['min'].append(min(epsiode_rew[-plot_of:]))


env.close()

plt.plot(cumu_epsiode_rew['ep'], cumu_epsiode_rew['avg'], label="average rewards")
plt.plot(cumu_epsiode_rew['ep'], cumu_epsiode_rew['max'], label="max rewards")
plt.plot(cumu_epsiode_rew['ep'], cumu_epsiode_rew['min'], label="min rewards")
plt.legend(loc=1)
plt.show()

style.use('ggplot')


    

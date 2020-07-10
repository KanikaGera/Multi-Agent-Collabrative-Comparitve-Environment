#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# In[1]:



# The environment is already saved in the Workspace and can be accessed at the file path provided below. 

# In[2]:


from unityagents import UnityEnvironment
import numpy as np
import pickle

env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# In[5]:


from ddpg_agent import Agent
agent=[]
for i in range(2):
    agent.append(Agent(state_size=48, action_size=action_size, random_seed=4))


# In[6]:


from collections import deque
import torch


# In[7]:


def ddpg(num_agents=2, n_episodes=20000, max_t=1000):

    episode_scores = []                                    # list containing scores from each episode
    scores_window = deque(maxlen=100)
    high= False
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current state (for each agent)
        states = np.reshape(states, (1, 48)) 
        
        for i_agent in range(num_agents):
                agent[i_agent].reset()

        score = np.zeros(num_agents)                       # initialize the score (for each agent)
        
        for t in range(max_t):
            actions=np.zeros((2,action_size))
            
            for i in range(2):
                actions[i] = agent[i].act(states)                    # select an action (for each agent):

            actions_ = np.reshape(actions, (1, 4))
            
            env_info = env.step(actions_)[brain_name]       # send all actions to tne environment
            
            next_states = env_info.vector_observations     # get next state (for each agent)
            next_states = np.reshape(next_states, (1, 48)) 
            
            rewards = env_info.rewards                     # get reward (for each agent)
            dones = env_info.local_done                    # see if episode finished

            for i_agent in range(num_agents):
                agent[i_agent].step(states,
                           actions[i_agent],
                           rewards[i_agent],
                           next_states,
                           dones[i_agent], t)              # update the system

            score += rewards                               # update the score (for each agent)
            states = next_states                           # roll over states to next time step
            
            if np.any(dones):                              # exit loop if episode finished
                break
        scores_window.append(np.max(score))
        episode_scores.append(np.max(score))

        if (i_episode%100==0):
            print('\rEpisode {}\tAverage Score: {:.4f}\t Max Score: {:.4f}'.format(i_episode, np.mean(scores_window), np.max(score)))
            torch.save(agent[0].actor_local.state_dict(), 'saved/checkpoint_actor_local_1.pth')
            torch.save(agent[0].critic_local.state_dict(), 'saved/checkpoint_critic_local_1.pth')
            torch.save(agent[0].actor_target.state_dict(), 'saved/checkpoint_actor_target_1.pth')
            torch.save(agent[0].critic_target.state_dict(), 'saved/checkpoint_critic_target_1.pth')
            torch.save(agent[1].actor_local.state_dict(), 'saved/checkpoint_actor_local_2.pth')
            torch.save(agent[1].critic_local.state_dict(), 'saved/checkpoint_critic_local_2.pth')
            torch.save(agent[1].actor_target.state_dict(), 'saved/checkpoint_actor_target_2.pth')
            torch.save(agent[1].critic_target.state_dict(), 'saved/checkpoint_critic_target_2.pth')
        if np.mean(scores_window) > 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent[0].actor_local.state_dict(), 'saved/solution_actor_local_1.pth')
            torch.save(agent[0].critic_local.state_dict(), 'saved/solution_critic_local_1.pth')
            torch.save(agent[0].actor_target.state_dict(), 'saved/solution_actor_target_1.pth')
            torch.save(agent[0].critic_target.state_dict(), 'saved/solution_critic_target_1.pth')
            torch.save(agent[1].actor_local.state_dict(), 'saved/checkpoint_actor_local_2.pth')
            torch.save(agent[1].critic_local.state_dict(), 'saved/checkpoint_critic_local_2.pth')
            torch.save(agent[1].actor_target.state_dict(), 'saved/checkpoint_actor_target_2.pth')
            torch.save(agent[1].critic_target.state_dict(), 'saved/checkpoint_critic_target_2.pth')
            high=True
            with open('saved/scores.list', 'wb') as scores_file:
                pickle.dump(episode_scores, scores_file)
        elif high:
            break
        if np.mean(scores_window) > 0.9:
            break
    return episode_scores


# In[ ]:


scores = ddpg()


# In[ ]:

#In[ ]:


# import random
# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# # In[ ]:


# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()


# In[ ]:


env.close()


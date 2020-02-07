
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from dqn_agent import Agent


env = UnityEnvironment(file_name = "Banana.app")
agent = Agent(state_size = 37, action_size = 4, seed = 0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth', map_location = lambda storage, loc: storage))

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode = False)[brain_name]
state = env_info.vector_observations[0]
score = 0
while True:         
        action = agent.act(state)                 # select an action
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
print(score)           
env.close()


from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from dqn_agent import Agent

# Initialise the environment:

# env = UnityEnvironment(file_name="Banana.app")
env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
agent = Agent(state_size=37, action_size=4, seed=0, name="base")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


def dqn(n_episodes=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, checkpoint='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        checkpoint (string): saved pytorch checkpoint for testing the agent
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment

        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score
        while True:
            action = agent.act(state, eps).astype(np.int32)  # select an action
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done, "ddqn")
            score += reward  # update the score
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format
                  (i_episode - 100, np.mean(scores_window)))
            torch.save(agent.q_network_local.state_dict(), checkpoint)
            break
    return scores


# Main Function:

if __name__ == '__main__':
    dqn_scores = dqn(n_episodes=500, eps_end=0.01, eps_decay=0.99, checkpoint='checkpoint.pth')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(dqn_scores)), dqn_scores)
    plt.title('Progress of the agent over the episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from dqn_agent import Agent

from memory_profiler import profile


class Trainer:
    def __init__(self):

        self.env = UnityEnvironment(file_name="VisualBanana_Windows_x86_64/Banana.exe")
        self.agent = Agent(state_size=37, action_size=4, seed=0, name="pixels")
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.last_states = deque(maxlen=3)
        for i in range(9):
            self.last_states.append(np.zeros((84, 84)))

    def get_state(self, frame, t):
        # make image in grayscale:
        frame = np.dot(frame[0], [0.2989, 0.5870, 0.1140])

        self.last_states.append(frame)

        if t > 2:
            state = np.stack((self.last_states[0], self.last_states[1], self.last_states[2], frame), axis=0)
        else:
            state = np.stack([frame for i in range(4)])
        return state

    @profile
    def dqn(self, n_episodes, checkpoint, eps_start=1., eps_end=0.1, eps_decay=0.995, alg="ddqn"):

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
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment

            state = self.get_state(env_info.visual_observations[0], 0)  # get the current state
            score = 0
            for t in range(300):
                action = self.agent.act(state, eps).astype(np.int32)  # select an action
                env_info = self.env.step(action)[self.brain_name]  # send the action to the environment
                next_state = self.get_state(env_info.visual_observations[0], t)
                reward = env_info.rewards[0]  # get the reward
                done = env_info.local_done[0]  # see if episode has finished
                # get the next state
                self.agent.step(state, action, reward, next_state, done, alg)
                score += reward  # update the score
                state = next_state  # roll over the state to next time step
                if done:  # exit loop if episode finished
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            eps = max(eps_end, eps_decay * eps)  # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if np.mean(scores_window) >= 13.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format
                      (i_episode - 100, np.mean(scores_window)))
                torch.save(self.agent.q_network_local.state_dict(), checkpoint)
                break
        return scores


# Main Function:

if __name__ == '__main__':
    dqn_scores = Trainer().dqn(n_episodes=700, checkpoint='checkpoint_pix.pth')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(dqn_scores)), dqn_scores)
    plt.title('Progress of the agent over the episodes')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

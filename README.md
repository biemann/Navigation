[//]: # (Image References)

[image2]: https://github.com/biemann/Navigation/blob/master/bin/209.png "Scores per Episode"
[image3]: https://github.com/biemann/Navigation/blob/master/bin/solved_for_15.png "Top Score"
[image4]: https://github.com/biemann/Navigation/blob/master/bin/mygif.gif "My Gif"

# Navigation

The first two sections are similar to the ones described in the project description. In our case, the agent has been trained using Mac OSX, so you may have to change the path dependencies in our code, following the instructions in the notebook https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb. We recommend following the instructions there and run the notebook to test whether the enviroment has been properly installed.

The report describing our experiments and the architecture lies further down in the Readme file.

## Introduction

For this project, the objective is to train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
2. Follow the instructions in https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb and make the notebook run.


## Training an Agent from scratch

If you want to train an agent, move inside the downloaded repository and type 

```
python train.py
```
in the terminal.

We can expect to solve the episode in 200-250 episodes using the original hyperparameters and network architecture. If you want 
to test other parameters, you would have to change the parameters at the top of the respective classes.

## Testing the Agent

In order to see a trained agent, you can type in the terminal

```
python test.py
```
We added two checkpoints, both located in the bin folder: `checkpoint.pth`, that is the result we get after having solved the task and `checkpoint_top.pth`that corresponds to the checkpoint of an agent after having reached an average score of 15.0 over the last 100 episodes.
## The Q-learning algorithm

The implementation follows very closely an example of a previous exercice in the Udacity Deep Reinforcement Learning Nanodegree, aiming to solve the Lunar Landing task of the OpenAI gym. The algorithm is an implementation of the Q-learning algorithm of the following breakthrough paper in the area : https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf. The project structure is therefore very similar to this exercice and most of the work of this project was to adapt the algorithm to this particular environment and optimise the hyperparameters. 

The Q-learning algorithm aims to approximate the optimal action-value function using a neural network. This extends the classical reinforcment learning approaches based on a look-up table to more complex problems such as this one. The most important idea of the paper was too use a replay buffer to collect the previous experiments. Then we give the network randomised samples of this buffer. This helps the learning of the algorithm a lot, as two subsequent experiments are highly correlated. In order to help the stability, the algorithm separates the network in two: a local network and a target network. We update the target network every 4 steps in our case. This hepls to combat oscillations and divergence when compared to online Q-learning.

## Implementation of the Q-learning algorithm

The `dqn_agent.py`class is an implementation of the Q-learning algorithm following the DeepMind paper as exposed in this nanodegree. The only slight modification we did was an implementation of a learning rate scheduler in order to help solving the task faster. 

The `model.py`is the neural network the Q-learning algorithm uses to make the agent learn. It uses three fully-connected layers of 128 neurons each. Hence, the model is slightly larger than the one used in the solution of the Lunar Landing exercice. We further added Batch Normalisation and Dropout layers to combat overfitting.

The `train.py`is the main function of our project. It adapts the Q-learning algorithm to this particular environment following an epsilon-greedy policy. The code follows the notebook https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb.

## Optimisation of the hyperparameters

We optimised the network architecture and the parameters in order to solve the task in the fewest episodes as possible. This arrives at the cost of robustness and it is possible that changing some hyperparameters even slightly may train agents that fail to solve the task. We will explain some experiments we made and how they affected the results.

The choice of the network architecture was particularily important for us. We experimented with 64 and 128 neurons per layer. We decided that all the layers should have the same number of neurons, following the practice of the state-of-the-art networks in computer vision. The 128 neuron-architecture takes more time to learn something: the 100 first episodes, the agent fails to understand the task and progresses far slower than using a 64-neuron architecture. However, it learns far faster once it learned something, enabling quite fast improvements after that. Hence, we decided to opt to the bigger architecture. We opted also for batch normalisation and dropout as overfitting becomes more present when having more neurons. We did not test more complicated activation functions, such as selu, different initialisations and other optimisers. 

One of the biggest issues in the training with this architecture was that the agent learned very slowly at the beginning and reached a plateau at the end, often stagnating at an average score of 11-12. That is why we opted for a learning rate scheduler, that lowered the learning rate by half every 100 episodes. The original learning rate of 5e-4 is relatively large in comparaision. We used a batch size of 128 and experience replay as described in the Atari paper. 

We also experienced with tau, gamma and epsilon. However, we only did slight changes to them, like lowering the epsilon decay to 0.99 because of the fast convergence of the algorithm. The other changes significantly decreased the model's efficiency.

## Results

We think that we were able to solve the task relatively fast, when compared to the benchmarks in the project description. As an example, we show the progress of an agent that has solved the task in 209 episodes (it took 309 episodes to reach an average score of 13 over the last 100 episodes).

![Scores per Episode][image2]

Due to the stochasticity of the network, it took once around 270 episodes to solve the task using the same hyperparameters. 

We experimented using this architecture to achieve high average rewards. Our highest result was 15. We would neet to adapt the parameters to achieve good results here. The epsilon decay of 0.99 is probably too fast (0.995 would likely give better results) and the learning rate decay would need to be adapted in order to achieve results of 16 or higher. We observed that with these parameters, the algorithm was becoming worse with time.

![Top Score][image3]

As an example on how the agent works, we made the following gif (we are sorry for the bad quality):

![My Gif][image4]

It achieved a score of 15. We found it interesting that it collected a blue banana in order to collect three yellow bananas in a short amount of time.

## Future Work

We would like to investigate the learning only the pixel values, as done in the original Q-learning paper. This will simulate the human behaviour in a closer way and is also essential in many applications, such as self-driving cars or robotics, where we would use a camera to watch the environment and would not have access to the additional data.

In this work specifically, we optimised the hyperparameters in order to solve the problem as fast as possible. However, we would need another architecture to achieve the best possible performance. This would certainly be an interesting task as well. The robustness of the algorithm and the study of domain shifts will certainly be of interest here. Note that the actual implementation is not reliable at all.

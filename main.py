import numpy as np
import random
import gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if __name__ == "__main__":
    env = gym.make("RL_env-0")

    num_episodes = 1000
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    learning_rate = 0.01
    # discount factor used in bellman equation
    gamma = 0.99
    # how frequently update the target network's weights
    target_update = 10
    # capacity of ReplayMemory
    memory_size = 10000
    batch_size = 4


    # set pytorch device tell pytorch use the GPU if it's available otherwise use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set env using the RLenv class
    env = RLenv(device)
    # set stratagy to be an instance of the EpsilonGreedyStrategy class
    stratagy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(stratagy, num_actions, device)
    memory = ReplayMemory(memory_size)
    # creating policy_net and target_net by creating 2 instances of DQN class
    # and pass in the input_state we get from env
    # put these networks on our defined device by pytorch's to function
    policy_net = DQN(env.input_state).to(device)
    target_net = DQN(env.input_state).to(device)
    # set the weights of target_net to be the same as those in the policy_net
    target_net.load_state_dict(policy_net.state_dict())
    # set target_net into eval mode(not in traning mode)
    target_net.eval()
    # set optimizer equal to the Adam optimizer which accepts our policy_net parameters
    # as those will be optimizing and are defined at learning rate
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)



    episode_durations = []
    # traning loop, iterate each episode
    for episode in range(num_episodes):
        env.reset()
        state = np.random.rand(0,3,size=3)

        # nested for loop that will iterate each time step in each episode
        for timestep in count():
            # for each time step, agent select action based on the current state and policy_net
            # use policy_net to select action if agent exploit the env rather than exploration
            action = agent.select_action(input_state, policy_net)
            reward, next_state = env.step(self, action)
            # then create an experience and put it onto the memory
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            # check if we can get a sample from memry to train our policy_net
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                # extract all states, actions, next_states, rewards from a given experience batch
                states, actions, next_states, rewards = extract_tensor(experiences)

                # Q values for the corresponding state action pairs that exteacted from experiences batch
                current_q_values = QValues.get_current(policy_net,input_states,actions)
                # Maximum Q values for the next_states in the batch using the target_net
                next_q_values = QValues.get_next(target_net, next_input_states)
                # calculate the target Qvalues
                target_qvalues = (next_q_values * gamma) + rewards
                # calculate the loss between the current QValues and the target QValues using mean square error as loss function
                loss = F.mse_loss(current_q_values, target_qvalues.unsqueeze(1))
                optimizer.zero_grad()
                # computes the gradient of the loss w.r.t all the weights and biases in the policy_net
                loss.backeard()
                # updates the weights and biases with the gradient computed above
                optimizer.step()

            if env.done:
                # if the episode is ended append current timestep to the episode_durations list
                # to store how long this episode lasted
                episode_durations.append(timestep)
                break

        # check if we should update the weights of target network before starting the new episode
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # the wole process end once it reach the number of episode
    env.close()

# Q value calculator
class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        return target_net(next_states).max(dim = 1)[0].detach()

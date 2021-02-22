import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
#from functions import frame_to_tensor
import random
import numpy as np

class ReplayMemory(object):
    """Replay Memory that stores the last size=1,000,000 transitions"""
    def __init__(self, size=1000000, frame_height=210, frame_width=160, 
                 agent_history_length=4, batch_size=32):
        """
        Args:
            size: Integer, Number of stored transitions
            frame_height: Integer, Height of a frame of an Atari game
            frame_width: Integer, Width of a frame of an Atari game
            agent_history_length: Integer, Number of frames stacked together to create a state
            batch_size: Integer, Number if transitions returned in a minibatch
        """
        self.size = size
        #self.frame_height = frame_height
        #self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0
        
        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        #self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.states = [None]*self.size
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        
        # Pre-allocate memory for the states and new_states in a minibatch
        #self.states = np.empty((self.batch_size, self.agent_history_length, 
        #                        self.frame_height, self.frame_width), dtype=np.uint8)
        #self.new_states = np.empty((self.batch_size, self.agent_history_length, 
        #                            self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)
        
    def add_experience(self, action, state, reward, terminal):
        """
        Args:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (210, 160) frame of an Atari game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        #if frame.shape != (self.frame_height, self.frame_width):
        #    raise ValueError('Dimension of frame is wrong!')
        #print("Add frame at position: ", self.current)
        self.actions[self.current] = action
        self.states[self.current] = state
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size
        #print("New current: ", self.current)
             
    def _get_state(self, index):
        if self.count == 0:
            raise ValueError("The replay memory is empty!")
        if index < self.agent_history_length - 1:
            raise ValueError("Index must be min 3")
        return self.states[index-self.agent_history_length+1:index+1, ...]
        
    def _get_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index < self.agent_history_length:
                    continue
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index
            
    def get_minibatch(self):
        """
        Returns a minibatch of self.batch_size = 32 transitions
        """
        if self.count < self.agent_history_length:
            raise ValueError('Not enough memories to get a minibatch')
        
        self._get_valid_indices()
            
        for i, idx in enumerate(self.indices):
            self.states_return[i] = self._get_state(idx - 1)
            self.new_states[i] = self._get_state(idx)
        
        return np.transpose(self.states_return), self.actions[self.indices], \
        self.rewards[self.indices], np.transpose(self.new_states), \
        self.terminal_flags[self.indices]

 

class DQN(nn.Module):
    """
    Main modell class. First 4 layers are convolutional layers, after that the model is split into the
    advantage and value stream. See the documentation. The convolutional layers are initialized with Kaiming He initialization.
    """
    def __init__(self, n_actions, in_channels, hidden=128):
        """
        Args:
            n_actions: Integer, amount of possible actions of the specific environment
            hidden: Integer, amount of hidden layers (To Do, hidden can change but split_size won't fit anymore)
        """
        super(DQN, self).__init__()
        
        self.n_actions = n_actions
        self.hidden = hidden
        self.in_channels = in_channels
        # Output of the 4th conv layer is 20480, if hidden is 128

        # Conv layers, initialize weights with Kaiming He initialization
        #self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, bias=False)
        #nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        #nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        #nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        #self.conv4 = nn.Conv2d(64, self.hidden, kernel_size=7, stride=1, bias=False)
        #nn.init.kaiming_uniform_(self.conv4.weight, nonlinearity='relu')

        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=32, kernel_size=2, stride=4)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.conv2 = nn.Conv3d(32, 64, kernel_size=1, stride=2)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.conv3 = nn.Conv3d(64, self.hidden, kernel_size=1, stride=1)
        nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')


        out = self.conv3(self.conv2(self.conv1(torch.zeros(1,100,3,3,3))))
        out = out.view(out.size(0), -1)
        self.split_size = int(out.size(1)/2)


        #Advantage and Value layer output
        self.advantage_l = torch.nn.Linear(self.split_size, self.n_actions)
        nn.init.kaiming_uniform_(self.advantage_l.weight, nonlinearity='relu')
        self.advantage_l.bias.data.zero_()

        self.value_l = torch.nn.Linear(self.split_size, 1)
        nn.init.kaiming_uniform_(self.value_l.weight, nonlinearity='relu')
        self.value_l.bias.data.zero_()



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        self.valuestream, self.advantagestream = torch.split(x, self.split_size, dim=1)
        
        self.advantage = self.advantage_l(self.advantagestream)
        self.value = self.value_l(self.valuestream)

        return self.advantage, self.value

    def predict_q(self, x):
        advantage, V_of_s = self(x)

        self.q_values = V_of_s + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_values


    def predict_action(self, x):
        q_values = self.predict_q(x)
        self.best_action = torch.argmax(q_values, 1)
        return self.best_action



class Agent():
    """
    The main agent that is optimized throughout the training process. The class consists of the two models (main and target),
    the optimizer and the memory.
    Args:
        n_actions: Integer, number of possible actions for the environment
        device: PyTorch device to which the models are sent to
        hidden: Integer, amount of hidden neurons in the model
        learning_rate: Float, learning rate for the training process
        gamma: Float, [0:1], discount factor for the loss function
        batch_size: Integer, specify size of each minibatch to process while learning
        agent_history_length: Integer, amount of stacked frames forming one transition
        memory_size: Integer, size of the replay memory
    """
    def __init__(self, n_actions, device, inp_size, hidden=128, learning_rate=0.0000625,
                 gamma=.99, batch_size=32, agent_history_length=4, memory_size=1000000):

        self.n_actions = n_actions
        self.device = device
        self.inp_size = inp_size
        self.hidden = hidden
        self.lr = learning_rate
        self.batch_size = batch_size
        self.agent_history_length= agent_history_length
        self.gamma = gamma
        self.memory_size = memory_size
  
        # Create 2 models
        self.main_dqn = DQN(self.n_actions, self.inp_size, self.hidden)
        self.target_dqn = DQN(self.n_actions, self.inp_size, self.hidden)
        # and send them to the device
        self.main_dqn = self.main_dqn.to(self.device)
        self.target_dqn = self.target_dqn.to(self.device)
        
        # Copy weights of the main model to the target model
        self.target_dqn.load_state_dict(self.main_dqn.state_dict())
        # and freeze target model. The model will be updated every now an then (specified in main function) 
        self.target_dqn.eval()

        
        self.replay_memory = ReplayMemory(size=self.memory_size, agent_history_length=self.agent_history_length, batch_size=self.batch_size, frame_height=210, frame_width=160)
        self.optimizer = torch.optim.Adam(self.main_dqn.parameters(), self.lr)

    def optimize(self):
        """
        Optimize the main model.
        Returns:
            Float, the loss between the predicted Q values from the main model and the target Q values from the target model
        """
        # get a minibatch of transitions
        states, actions, rewards, new_states, terminal_flags = self.replay_memory.get_minibatch()
        #
        # states: (32, 4, 210, 160)
        # actions: (32, 9)
        # rewards: (32)
        # new_states: (32, 4, 210, 160)
        # terminal_flags: (32)
        #

        states_interpol = []
        next_states_interpol = []
        for i in range(self.batch_size):
            states_interpol.append(states[i].getValue())
            next_states_interpol.append(new_states[i].getValue())

        states_interpol = torch.stack(states_interpol).to(self.device)
        next_states_interpol = torch.stack(next_states_interpol).to(self.device)
        
        # states = torch.FloatTensor(states/255.).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.IntTensor(rewards).to(self.device)
        #new_states = torch.FloatTensor(new_states/255.).to(self.device)
        terminal_flags = torch.BoolTensor(terminal_flags).to(self.device)

        # predict the best actions for all the next states, shape: (32)
        arg_q_max = self.main_dqn.predict_q(new_states).max(1)[1]

        # predict the Q values for all the next states,
        # get the Q value of the target model for the predicted actions arg_q_max, shape: (32)
        double_q = self.target_dqn.predict_q(new_states).gather(1, arg_q_max.unsqueeze(-1)).squeeze(1).detach()

        # Belman equation. Make sure that if episode is over, target_q = rewards, shape: (32)
        target_q = rewards + (self.gamma * double_q * ~terminal_flags)

        # predict the q values for the current states,
        # select Q value of the best action that was performed for the current state, shape: (32)
        predict_Q = self.main_dqn.predict_q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        loss = F.smooth_l1_loss(input=predict_Q, target=target_q, reduction='mean')
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # delete all variables to ensure low memory usage
        del double_q
        del target_q
        del predict_Q
        
        return loss

class Action_Scheduler():
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""
    def __init__(self, num_actions, model, eps_initial=1, eps_final=0.1, eps_final_step=0.01,
                eps_annealing_steps=1000000, replay_memory_start_size=50000, 
                max_steps=25000000):
        """
        Args:
            num_actions: Integer, number of possible actions
            model: A DQN object
            eps_initial: Float, Exploration probability for the first 
                replay_memory_start_size frames
            eps_final: Float, Exploration probability after 
                replay_memory_start_size + eps_annealing_frames frames
            eps_final_frame: Float, Exploration probability after max_frames frames
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_frames: Int, Number of frames over which the 
                exploration probabilty is annealed from eps_initial to eps_final
            replay_memory_start_size: Integer, Number of frames during 
                which the agent only explores
            max_frames: Integer, Total number of frames shown to the agent
        """
        self.num_actions = num_actions
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_step
        self.eps_annealing_frames = eps_annealing_steps
        self.replay_memory_start_size = replay_memory_start_size
        self.max_frames = max_steps
        self.model = model

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_memory_start_size
        self.slope_2 = - (self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_memory_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2 * self.max_frames

    def get_action(self, frame_number, state, evaluation=False):
        if evaluation:
            eps = 0.0
        elif frame_number < self.replay_memory_start_size:
            eps = self.eps_initial
        elif frame_number >= self.replay_memory_start_size and frame_number < self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope * frame_number + self.intercept
        elif frame_number >= self.replay_memory_start_size + self.eps_annealing_frames:
            eps = self.slope_2 * frame_number + self.intercept_2

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.num_actions)

        if len(state) < 4:
            return np.random.randint(0, self.num_actions)
        else:
            return self.model.predict_action(state).item()
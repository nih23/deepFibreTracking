# Source: https://github.com/Curt-Park/rainbow-is-all-you-need
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from zmq import device

from tqdm import tqdm

from dfibert.ext.soft_dtw_cuda import SoftDTW
from dfibert.tracker import save_streamlines
from dfibert.tracker.nn.mlp import MLP
from dfibert.tracker.nn._segment_tree import MinSegmentTree, SumSegmentTree
from dfibert.util import set_seed

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99,
        device: torch.device = torch.device('cpu')
    ):
        self.obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32, device=device)
        self.acts_buf = torch.zeros([size], dtype=torch.long, device=device)
        self.rews_buf = torch.zeros([size], dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.float32,
        next_obs: torch.Tensor,
        done: torch.bool,
     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

   # def sample_batch(self) -> Dict[str, np.ndarray]:
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * ~d
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
        device: torch.device = torch.device('cuda')
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma, device
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.device = device
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.float32,
        next_obs: torch.Tensor,
        done: torch.bool,
     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition
   
    def sample_batch(self, beta: float = 0.4) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = torch.tensor([self._calculate_weight(i, beta) for i in indices], dtype=torch.float32, device=self.device)
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: torch.Tensor):    
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    
        
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = MLP(input_size=in_dim, output_size=512, hidden_size = 1024, num_hidden = 4)
        '''
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, in_dim*2), 
            nn.LeakyReLU(),
            nn.Linear(in_dim*2, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 1024), 
            nn.LeakyReLU(),
            nn.Linear(1024, 512), 
            nn.LeakyReLU(),
            nn.Linear(512, 512), 
            nn.LeakyReLU()
        )
        '''
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(512, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(512, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.leaky_relu(self.advantage_hidden_layer(feature))
        val_hid = F.leaky_relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        lr: float = 1e-3,
        adam_eps: float = 1.5e-4,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        wandb_log: bool = False,
        num_val_streamlines: int = 10,
        seed = 2342,
        path: str = './training'
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        set_seed(seed)
        
        obs_dim = env.reset().shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha
        self.memory_size = memory_size
        self.adam_eps = adam_eps
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = device

        # enable W&B logging
        self.wandb_log = wandb_log
        
        # fix seeds of streamlines for computation of validation loss
        self.val_seed_indices = np.random.randint(len(self.env.seeds), size=num_val_streamlines)
        
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=self.alpha, device=self.device
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma, device=self.device
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), self.lr, eps=self.adam_eps)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False
        self.step_offset = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        selected_action = torch.argmax(self.dqn(state))
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: torch.Tensor, backwards: torch.bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take an action and return the response of the env."""
        next_state, reward, done, _ = self.env.step(action, backwards)
        #### not needed if environment is moved to gpu
        if not torch.is_tensor(next_state):
            print("needed to tensor() next_state")
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(reward):
            reward = torch.tensor(reward, dtype=torch.int64, device=self.device)
        if not torch.is_tensor(done):
            done = torch.tensor(done, dtype=torch.bool, device=self.device)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = samples["weights"].reshape(-1, 1)
        indices = samples["indices"]
        
        # 1-step Learning loss
        #print("-- update model --")
        #print(samples)
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        #print("element wise loss: ", elementwise_loss)
        loss_for_prior = elementwise_loss
        new_priorities = loss_for_prior + self.prior_eps
        #print("new priorities: ", elementwise_loss)
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def pretrain(self, num_epochs: int = 1000, batch_size: int = 1024, seed_selection_fa_Threshold: float = 0.2, lr: float = 1e-4, path: str="./pretraining_checkpoints/"):
        from dfibert.tracker.nn.supervised_pretraining import train as pretraining
        self.is_test = True
        self.dqn = pretraining(dqn=self.dqn, env=self.env, epochs=num_epochs, batch_size=batch_size, seed_selection_fa_Threshold=seed_selection_fa_Threshold, lr=lr, path = path, wandb_log=False)
        #self.dqn = pretraining(dqn=self.dqn, env=self.env, epochs=num_epochs, batch_size=batch_size, seed_selection_fa_Threshold=seed_selection_fa_Threshold, lr=lr, path = path, wandb_log=self.wandb_log)
        self._target_hard_update()

        gt_distance, gt_reward, agent_reward = self._compare_to_gt()
        print("Finished pretraining the network. Mean distance to ground-truth streamlines: %.2f" % (gt_distance))
        print("Ground-truth reward: %.2f, Agent reward: %.2f" % (gt_reward, agent_reward))
        
        
    def train(self, num_steps: int, steps_done: int = 1, plotting_interval: int = 200, checkpoint_interval: int = 20000, path: str = "./", losses: list = [], scores: list = [], plot: bool = False):
        """Train the agent."""
        self.is_test = False

        if self.wandb_log:
            import wandb
            wandb.config.update = {
                        "max steps": num_steps,
                        "network_update_every": self.target_update,
                        "prior_epsilon": self.prior_eps,
                        "batch_size": self.batch_size,
                        "gamma": self.gamma,
                        "alpha": self.alpha,
                        "lr": self.lr,
                        "adam_eps": self.adam_eps,
                        "beta": self.beta,
                        "v_min": self.v_min,
                        "v_max": self.v_max,
                        "atom_size": self.atom_size,
                        "n_step": self.n_step,
                        "support": self.support,
                        "memory_size": self.memory_size,
                     }

            wandb.watch(self.dqn, log='all')
            wandb.watch(self.dqn_target, log='all')

        state = self.env.reset()
        backwards = False
        update_cnt = 0
        score = 0
        streamline_len = deque(maxlen=1000)
        cur_streamline_len = 0

        with tqdm(range(steps_done, num_steps + 1), unit="epochs", ascii=True) as pbar:
            for step_idx in pbar:
                action = self.select_action(state)
                next_state, reward, done = self.step(action, backwards)

                state = next_state
                score += reward.detach().cpu().numpy()

                # NoisyNet: removed decrease of epsilon

                # PER: increase beta
                fraction = min(step_idx / num_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # if episode ends
                if done:
                    backwards = not backwards
                    if backwards:
                        seed_index = self.env.seed_index
                    else:
                        seed_index = None
                    # TODO: If env is moved to GPU, the following line can be changed to
                    state = self.env.reset(seed_index)
                    #state = torch.tensor(self.env.reset(seed_index), dtype=torch.float32, device=self.device)
                    scores.append(score)
                    streamline_len.append(cur_streamline_len)
                    if self.wandb_log:
                        wandb.log({
                            'Mean episode reward over past 1000 episodes': np.mean(scores[-1000:]),
                            'Median episode reward over past 1000 episodes': np.median(scores[-1000:]),
                            'Mean streamline length over past 1000 episodes': np.mean(list(streamline_len)),
                            'Median streamline length over past 1000 episodes': np.median(list(streamline_len))
                             }, 
                            step=step_idx)
                    score = 0
                    cur_streamline_len = 0
                    pbar.set_postfix(step=step_idx, reward=np.median(scores[-1000:]),length=np.median(list(streamline_len)))

                cur_streamline_len += 1

                # if training is ready
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    if self.wandb_log:
                        wandb.log({'Loss': loss}, step=step_idx)
                    losses.append(loss)
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # plotting
                if plot and step_idx % plotting_interval == 0:
                    self._plot(step_idx, scores, losses)

                if step_idx % checkpoint_interval == 0 and step_idx != steps_done:
                    #print("Step number: ", step_idx, "Avg. reward: ", np.mean(scores[-1000:]))
                    print("Step number: ", step_idx, "Mean reward: ", np.mean(scores[-1000:]))
                    self._save_model(path, step_idx, scores, losses, num_steps)

                    print("Further evaluation on 20 random streamlines:")
                    mean_distance_to_gt, mean_gt_reward, mean_agent_reward = self._compare_to_gt()
                    print("Mean distance to gt streamlines: ", mean_distance_to_gt)
                    print("Mean gt reward:  ", mean_gt_reward)
                    print("Mean agent reward: ", mean_agent_reward)
                    if self.wandb_log:
                        wandb.log({"Mean distance to gt streamlines": mean_distance_to_gt}, step=step_idx)
                        wandb.log({"Mean gt reward": mean_gt_reward}, step=step_idx)
                        wandb.log({"Mean agent reward": mean_agent_reward}, step=step_idx)
                
        #self.env.close()

    def _compare_to_gt(self):
        #gt_streamlines = []
        #agent_streamlines = []
        distances = []
        gt_rewards = []
        agent_rewards = []
        sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        
        print("Comparing to ground-truth streamlines...")
        for i in tqdm(self.val_seed_indices, ascii=True):
            _ = self.env.reset(seed_index=i)
            seed_point = self.env.state.getCoordinate().detach().cpu().tolist()
            # track ground-truth streamline and predicted streamline by agent
            gt_streamline, gt_reward = self.env._track_single_streamline(i)
            agent_streamline, agent_reward = self.env._track_single_streamline(i, self.select_action)
            gt_rewards.append(gt_reward.item())
            agent_rewards.append(agent_reward.item())
            
            gt_streamline = torch.stack(gt_streamline).unsqueeze(0) # 1 x noPts x 3
            agent_streamline = torch.stack(agent_streamline).unsqueeze(0) #1 x noPts x 3
            distance = sdtw(gt_streamline, agent_streamline) 
            distances.append(distance.item())

        # return the mean of all l2 distances, ground-truth and agent streamline rewards
        return np.mean(distances), np.mean(gt_rewards), np.mean(agent_rewards)

                
    def test(self) -> List[np.ndarray]:
        raise NotImplementedError
        # """Test the agent."""
        # self.is_test = True
        
        # state = self.env.reset()
        # done = False
        # score = 0
        
        # steps = []
        # while not done:
        #     steps.append(self.env.render(mode="rgb_array"))
        #     action = self.select_action(state)
        #     next_state, reward, done = self.step(action)

        #     state = next_state
        #     score += reward
        
        # print("score: ", score)
        # self.env.close()
        
        # return steps

    # def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
    def _compute_dqn_loss(self, samples: Dict[str, torch.Tensor], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = samples["acts"]
        reward = samples["rews"].reshape(-1,1)
        done = samples["done"].reshape(-1, 1)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]
            t_z = reward + (1 - done) * gamma * self.support.unsqueeze(0)

            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        step_idx: int, 
        scores: List[float], 
        losses: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('step %s. avg. score: %s' % (step_idx, np.median(scores[-1000:])))
        plt.plot(scores)
        plt.ylabel('Episode reward')
        plt.xlabel("No. episodes")
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.ylabel('Loss')
        plt.xlabel("No. updates")
        plt.show()

    def _save_model(self, path, num_steps, rewards, losses, max_steps):
        path = path + '/checkpoints/'
        os.makedirs(path, exist_ok=True)
        path = path + 'rainbow_%d_%.2f.pth' % (num_steps, np.mean(rewards[-1000:]))
        print("Writing checkpoint to %s" % (path))
        checkpoint = {}
        checkpoint["num_steps"] = num_steps
        checkpoint["rewards"] = rewards
        checkpoint["losses"] = losses
        checkpoint["max_steps"] = max_steps
        checkpoint["network_update_every"] = self.target_update
        checkpoint["network"] = self.dqn.state_dict()
        checkpoint["prior_epsilon"] = self.prior_eps
        checkpoint["batch_size"] = self.batch_size
        checkpoint["learning_rate"] = self.lr
        checkpoint["adam_eps"] = self.adam_eps
        checkpoint["gamma"] = self.gamma
        checkpoint["alpha"] = self.alpha
        checkpoint["beta"] = self.beta
        checkpoint["v_min"] = self.v_min
        checkpoint["v_max"] = self.v_max
        checkpoint["atom_size"] = self.atom_size
        checkpoint["n_step"] = self.n_step
        checkpoint["support"] = self.support
        checkpoint["memory_size"] = self.memory_size
        torch.save(checkpoint, path)

    def _load_model(self, path):
        print("Loading checkpoint file %s" % (path))
        checkpoint = torch.load(path)
        num_steps = checkpoint["num_steps"]
        rewards = checkpoint["rewards"]
        losses = checkpoint["losses"]
        max_steps = checkpoint["max_steps"]
        self.target_update = checkpoint["network_update_every"]
        self.dqn.load_state_dict(checkpoint["network"])
        self.dqn_target.load_state_dict(checkpoint["network"])
        self.prior_eps = checkpoint["prior_epsilon"] 
        self.batch_size = checkpoint["batch_size"]
        self.lr = checkpoint["learning_rate"]
        self.adam_eps = checkpoint["adam_eps"]
        self.gamma = checkpoint["gamma"]
        self.alpha = checkpoint["alpha"]
        self.beta = checkpoint["beta"]
        self.v_min = checkpoint["v_min"]
        self.v_max = checkpoint["v_max"]
        self.atom_size = checkpoint["atom_size"]
        self.n_step = checkpoint["n_step"]
        self.support = checkpoint["support"]
        self.memory_size = checkpoint["memory_size"]

        return num_steps, rewards, losses, max_steps

    def resume_training(self, path, plot: bool = False, checkpoint_interval: int = 2000):
        num_steps, rewards, losses, max_steps = self._load_model(path)
        #remaining_steps = max_steps - num_steps   # set max_steps to remaining amount of steps

        path_dir = os.path.dirname(path)        # get the base directory 
        path_dir = os.path.split(path_dir)[0]

        print("Resume training at %d / %d steps." % (num_steps, max_steps) )
        self.train(num_steps=max_steps, steps_done=num_steps, losses = losses, scores = rewards, path=path_dir, checkpoint_interval=checkpoint_interval, plot=plot)

    def create_tractogram(self, path: str = './'):
        self.is_test = True                                                 

        # track and convert into RAS
        streamlines = self.env.track(agent=self.select_action)
        streamlines_ras = [self.env.dataset.to_ras(torch.stack(sl).cpu()) for sl in streamlines]
        
        # store streamlines as vtk file
        save_streamlines(streamlines=streamlines_ras, path=path)
        
        return streamlines

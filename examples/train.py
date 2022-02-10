import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import argparse
import random

import os, sys
sys.path.insert(0,'..')

from dfibert.tracker.nn.rainbow_agent import DQNAgent

import dfibert.envs.RLTractEnvironment_fast as RLTe


def train(path, max_steps=3000000, batch_size=32, replay_memory_size=20000, gamma=0.99, network_update_every=10000, learning_rate=0.0000625, checkpoint_every=200000):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Init environment..")
    
    seeds_CST = np.load('data/ismrm_seeds_CST.npy')
    seeds_CST = torch.from_numpy(seeds_CST)

    env = RLTe.RLTractEnvironment(dataset = 'ISMRM', step_width=0.2,
                                  device = device, seeds = seeds_CST, action_space=20,
                                  tracking_in_RAS = False, odf_state = False, odf_mode = "DTI")

    print("..done!")
    print("Init agent..")


    agent = DQNAgent(env=env, memory_size = replay_memory_size,
                    batch_size = batch_size,
                    target_update = network_update_every,
                    lr = learning_rate,
                    gamma = gamma,
                    device = device,
                    )
    print("..done!")
    print("Start training...")

    agent.train(num_steps = max_steps, checkpoint_interval=checkpoint_every, path = path, plot=False)
    


def resume(path, max_steps=3000000, batch_size=32, replay_memory_size=20000, gamma=0.99, network_update_every=10000, learning_rate=0.0000625, checkpoint_every=200000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Init environment..")
    
    seeds_CST = np.load('data/ismrm_seeds_CST.npy')
    seeds_CST = torch.from_numpy(seeds_CST)

    env = RLTe.RLTractEnvironment(dataset = 'ISMRM', step_width=0.2,
                                  device = device, seeds = seeds_CST, action_space=20,
                                  tracking_in_RAS = False, odf_state = False, odf_mode = "DTI")

    print("..done!")
    print("Init agent..")

    agent = DQNAgent(env=env, memory_size = replay_memory_size,
                    batch_size = batch_size,
                    target_update = network_update_every,
                    lr = learning_rate,
                    gamma = gamma,
                    device = device,
                    )

    print("..done!")
    print("Resume training..")

    agent.resume_training(path=path, plot=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", default=3000000, type=int, help="Choose maximum amount of training steps")
    #parser.add_argument("--start_learning", default=2000, type=int, help="Set amount of steps after which epsilon will be decreased and the agent will be learning")
    parser.add_argument("--replay_memory_size", default=100000, type=int, help="Set amount of past expiriences stored in the replay memory")
    #parser.add_argument("--eps_annealing_steps", default=100000, type=int, help="Set amount of steps after which epsilon is decreased more slowly until max_steps")
    #parser.add_argument("--agent_history_length", default=1, type=int, help="Choose how many past states are included in each input to update the agent")
    #parser.add_argument("--evaluate_every", default=20000, type=int, help="Set evaluation interval")
    #parser.add_argument("--eval_runs", default=10, type=int, help="Set amount of runs performed during evaluation")
    parser.add_argument("--network_update_every", default=1000, type=int, help="Set target network update frequency")
    #parser.add_argument("--max_episode_length", default=550, type=int, help="Set maximum episode length")
    parser.add_argument("--batch_size", default=32, type=int, help="Set batch size retrieved from memory for learning")
    parser.add_argument("--learning_rate", default=0.0000625, type=float, help="Set learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Set discount factor for Bellman equation")
    #parser.add_argument("--eps_final", default=0.1, type=float, help="Set first value to which epsilon is lowered to after eps_annealing_steps")
    #parser.add_argument("--eps_final_step", default=0.01, type=float, help="Set the second value to which epsilon is lowered to from eps_final until max_steps")
    parser.add_argument("--checkpoint_every", default=200000, type=int, help="Set checkpointing interval")

    parser.add_argument("--path", default=".", type=str, help="Set default saving path of logs and checkpoints")

    parser.add_argument("--resume_training", dest="resume", action='store_true', help="Load checkpoint from path folder and resume training")
    parser.add_argument("--wandb", action='store_true', help="Log training on W&B")
    #parser.add_argument("--odf-as-state-value",dest="odf_state", action='store_true')
    #parser.set_defaults(odf_state=False)

    args = parser.parse_args()

    if args.resume:
        import glob
        paths = glob.glob(args.path+'*.pt')
        p_cp = max(paths, key=os.path.getctime)

        resume(p_cp)

    else:
        train(args.path, max_steps=args.max_steps, replay_memory_size=args.replay_memory_size, 
              batch_size=args.batch_size, gamma=args.gamma, 
              network_update_every=args.network_update_every, learning_rate=args.learning_rate,
              checkpoint_every=args.checkpoint_every)
    
        

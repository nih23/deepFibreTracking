import torch
import numpy as np
import argparse

import os, sys
sys.path.insert(0,'..')

from dfibert.tracker.nn.rainbow_agent import DQNAgent
from dfibert.util import set_seed

import dfibert.envs.RLTractEnvironment_fast as RLTe


def train(path, pretraining=False, max_steps=3000000, batch_size=32, replay_memory_size=20000, gamma=0.99, network_update_every=10000, learning_rate=0.0000625, checkpoint_every=200000, wandb=False, step_width = 0.8, odf_mode = "CSD"):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Init environment..")
    
    #seeds_CST = np.load('data/ismrm_seeds_CST.npy')
    #seeds_CST = torch.from_numpy(seeds_CST)

    env = RLTe.RLTractEnvironment(dataset = 'ISMRM', step_width=step_width,
                                  device = device, seeds = None, action_space=20,
                                  tracking_in_RAS = False, odf_state = False, odf_mode = odf_mode)

    print("..done!")
    print("Init agent..")


    agent = DQNAgent(env=env, memory_size = replay_memory_size,
                    batch_size = batch_size,
                    target_update = network_update_every,
                    lr = learning_rate,
                    gamma = gamma,
                    device = device,
                    wandb_log=wandb
                    )
    print("..done!")

    if pretraining:
        print("Start pretraining..")
        agent.pretrain(path=path+'super_checkpoints/')

    print("Start DQL...")
    agent.train(num_steps = max_steps, checkpoint_interval=checkpoint_every, path = path, plot=False)
    


def resume(path, max_steps=3000000, batch_size=32, replay_memory_size=20000, gamma=0.99, network_update_every=10000, learning_rate=0.0000625, checkpoint_every=200000, wandb=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    print("Init environment..")
    
    #seeds_CST = np.load('data/ismrm_seeds_CST.npy')
    #seeds_CST = torch.from_numpy(seeds_CST)

    env = RLTe.RLTractEnvironment(dataset = 'ISMRM', step_width=0.2,
                                  device = device, seeds = None, action_space=20,
                                  tracking_in_RAS = False, odf_state = False, odf_mode = "CSD")

    print("..done!")
    print("Init agent..")

    agent = DQNAgent(env=env, memory_size = replay_memory_size,
                    batch_size = batch_size,
                    target_update = network_update_every,
                    lr = learning_rate,
                    gamma = gamma,
                    device = device,
                    wandb_log=wandb
                    )               

    print("..done!")
    print("Resume training..")

    agent.resume_training(path=path, plot=False, wandb=wandb)

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
    parser.add_argument("--batch_size", default=512, type=int, help="Set batch size retrieved from memory for learning")
    parser.add_argument("--learning_rate", default=0.0000625, type=float, help="Set learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Set discount factor for Bellman equation")
    #parser.add_argument("--eps_final", default=0.1, type=float, help="Set first value to which epsilon is lowered to after eps_annealing_steps")
    #parser.add_argument("--eps_final_step", default=0.01, type=float, help="Set the second value to which epsilon is lowered to from eps_final until max_steps")
    parser.add_argument("--checkpoint_every", default=200000, type=int, help="Set checkpointing interval")

    parser.add_argument("--path", default=".", type=str, help="Set default saving path of logs and checkpoints")
    parser.add_argument("--seed", default=42, type=int, help="Set a seed for the training run")

    parser.add_argument("--step_width", default=0.8, type=float, help="step width for tracking")
    parser.add_argument("--odf_mode", default="CSD", type=str, help="compute ODF in reward based on DTI or CSD?")
    
    parser.add_argument("--pretrain", action='store_true', help="Pretrain the DQN with superwised learnin")
    parser.add_argument("--resume_training", dest="resume", action='store_true', help="Load checkpoint from path folder and resume training")
    
    parser.add_argument("--wandb", action='store_true', help="Log training on W&B")
    parser.add_argument("--wandb_project", default="deepFibreTracking", type=str, help="Set name of W&B project")
    parser.add_argument("--wandb_entity", default=None, type=str, help="Set entity of W&B project")
    #parser.add_argument("--odf-as-state-value",dest="odf_state", action='store_true')
    #parser.set_defaults(odf_state=False)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.wandb:
        import wandb
        config = args

    if args.resume:
        if args.wandb:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, resume=True)
        resume(args.path, batch_size=args.batch_size, gamma=args.gamma, checkpoint_every=args.checkpoint_every, wandb=args.wandb)

    else:
        if args.wandb:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, resume='allow')
        train(args.path, pretraining=args.pretrain, max_steps=args.max_steps, replay_memory_size=args.replay_memory_size, 
              batch_size=args.batch_size, gamma=args.gamma, 
              network_update_every=args.network_update_every, learning_rate=args.learning_rate,
              checkpoint_every=args.checkpoint_every, wandb=args.wandb, step_width = args.step_width, odf_mode = args.odf_mode)
    
        

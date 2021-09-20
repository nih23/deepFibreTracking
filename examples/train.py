import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import argparse
import random

import os, sys
sys.path.insert(0,'..')

from dfibert.tracker.nn.rl import Agent, Action_Scheduler, DQN

import dfibert.envs.RLtractEnvironment as RLTe


def save_model(path_checkpoint, model, epoch, mean_reward, epsilon):
    print("Writing checkpoint to %s" % (path_checkpoint))
    checkpoint = {}
    checkpoint["model"] = model.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["mean_reward"] = mean_reward
    checkpoint["epsilon"] = epsilon
    torch.save(checkpoint, path_checkpoint)


def load_model(path_checkpoint):
    print("Loading checkpoint from %s" % (path_checkpoint))
    checkpoint = torch.load(path_checkpoint)
    #model.load_state_dict(checkpoint['model'])
    model = checkpoint['model']
    epoch = checkpoint['epoch']
    mean_reward = checkpoint['mean_reward']
    epsilon = checkpoint['epsilon']

    return model, epoch, mean_reward, epsilon


def train(path, eps=1.0, step_counter=0, max_steps=3000000, batch_size=32, replay_memory_size=20000, start_learning=10000, eps_annealing_steps=100000, eps_final=0.1, eps_final_step=0.01, gamma=0.99, agent_history_length=1, evaluate_every=20000, eval_runs=5, network_update_every=10000, max_episode_length=200, learning_rate=0.0000625, model=None):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device..", device)
    print("Init environment..")
    env = RLTe.RLtractEnvironment(stepWidth=1., action_space=20, maxL2dist_to_State=0.2, device = 'cpu', pReferenceStreamlines='data/HCP307200_DTI_min40.vtk')
    print("..done!")
    n_actions = env.action_space.n

    print("Init agent..")
    state = env.reset().getValue()
    agent = Agent(n_actions=n_actions, inp_size=state.shape, device=device, gamma=gamma, agent_history_length=agent_history_length, memory_size=replay_memory_size, batch_size=batch_size, learning_rate=learning_rate)
    if model != None:
        agent.main_dqn.load_state_dict(model)
        agent.target_dqn.load_state_dict(model)
    #print("Init epsilon-greedy action scheduler")
    #action_scheduler = Action_Scheduler(num_actions=n_actions, max_steps=max_steps, eps_annealing_steps=eps_annealing_steps, replay_memory_start_size=start_learning, model=agent.main_dqn)
    #step_counter = 0
    eps_rewards = []
    episode_lengths = []

    print("Start training...")
    while step_counter < max_steps:
        epoch_step = 0
        while (epoch_step < evaluate_every) or (step_counter < start_learning):
            # one step consists of two sub-steps:
            # 1 fill replay memory
            # 2 train agent
            
            # evaluate_every: evaluate accuracy of agent every `evaluate_every` steps
            # start_learning: fill memory for `start_learning` steps (experiences) before actual training of agent, 
            #                 should be equal or larger than batch size
            state = env.reset()
            episode_reward_sum = 0
            terminal = False
            episode_step_counter = 0
            positive_run = 0
            points_visited = 0
            
            negative_rewards = 0
            agent_train = False
            
            # reduce epsilon
            if step_counter > start_learning:
                eps = max(eps * 0.999, 0.01)
            
            action_hist = []
            
            # play an episode
            while episode_step_counter <= 1000.:
                # episode_step_counter denotes the number of steps that the agent is supposed to take
                # for a single episode (i.e. streamline) 

                ########################
                ## FILL REPLAY MEMORY ##
                ########################
                
                # get an action with epsilon-greedy strategy
                if random.random() < eps:
                    # random vs supervised action
                    action = np.random.randint(env.action_space.n)           # either random action
                    #action = env._get_best_action()                         # supervised learning
                else:   
                    # or take action from agent
                    agent.main_dqn.eval()
                    with torch.no_grad():
                        state_v = torch.from_numpy(state.getValue()).unsqueeze(0).float().to(device)
                        action = torch.argmax(agent.main_dqn(state_v)).item()
                    agent.main_dqn.train()
                
                # perform step on environment
                next_state, reward, terminal, _ = env.step(action)
                
                episode_step_counter += 1
                step_counter += 1
                epoch_step += 1
                
                episode_reward_sum += reward
                
                # store experience in replay buffer
                agent.replay_memory.add_experience(action=action, state = state.getValue(), reward=reward, new_state = next_state.getValue(), terminal=terminal)
                state = next_state
                                
                ########################
                ########################
                ########################
                
                
                # optimize agent after certain amount of steps
                if (step_counter > start_learning) and (step_counter % 4 == 0):
                    ### debugging optimization function
                    
                    states, actions, rewards, new_states, terminal_flags = agent.replay_memory.get_minibatch()

                    states = torch.from_numpy(states).to(device)
                    next_states = torch.from_numpy(new_states).to(device)
                    actions = torch.from_numpy(actions).unsqueeze(1).long().to(device)
                    rewards = torch.from_numpy(rewards).to(device)
                    terminal_flags = torch.from_numpy(terminal_flags).to(device)
                    
                    
                    state_action_values = agent.main_dqn(states).gather(1, actions).squeeze(-1)
                    next_state_actions = torch.argmax(agent.main_dqn(next_states), dim=1)
                    
                    action_hist.append(next_state_actions.detach().view(-1).cpu().numpy())

                    next_state_values = agent.target_dqn(next_states).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
                    # 
                    next_state_values[terminal_flags] = 0.0
                    #
                    expected_state_action_values = next_state_values.detach() * 0.9995 + rewards
                    #
                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
                    agent.optimizer.zero_grad()
                    loss.backward()
                    agent.optimizer.step()
                    
                # update target network after certain amount of steps    
                if step_counter > start_learning and step_counter % network_update_every == 0:
                    agent.target_dqn.load_state_dict(agent.main_dqn.state_dict())
                
                # if epsiode has ended, step out of the episode while loop
                if terminal:
                    break
                    
            # keep track of past episode rewards
            eps_rewards.append(episode_reward_sum)
            if len(eps_rewards) % 20 == 0:
                with open(path+'/logs/rewards.dat', 'a') as reward_file:
                    print("[{}] {}, {}".format(len(eps_rewards), step_counter, np.mean(eps_rewards[-100:])), file=reward_file)
                print("{}, done {} episodes, {}, current epsilon {}, av actions {}".format(step_counter, len(eps_rewards), np.mean(eps_rewards[-100:]), eps, np.mean(action_hist)))#action_scheduler.eps_current))
                p_cp = '%s/checkpoints/defi_%d_%.2f.pt' % (path, step_counter, np.mean(eps_rewards[-100:]))
                save_model(p_cp, agent.main_dqn, step_counter, np.mean(eps_rewards[-100:]), eps)
                
        ## evaluation        
        eval_rewards = []
        episode_final = 0
        agent.main_dqn.eval()
        for _ in range(eval_runs):
            eval_steps = 0
            state = env.reset()
            
            eval_episode_reward = 0
            negative_rewards = 0
            
            # play an episode
            while eval_steps < 1000:
                # get the action from the agent
                with torch.no_grad():
                        state_v = torch.from_numpy(state.getValue()).unsqueeze(0).float().to(device)
                        action = torch.argmax(agent.main_dqn(state_v)).item()
                    
                # perform a step on the environment
                next_state, reward, terminal, _ = env.step(action)
                
                eval_steps += 1
                
                eval_episode_reward += reward
                state = next_state
                
                # step out of the episode while loop if 
                if terminal:
                    terminal = False
                    if reward == 1.:
                        episode_final += 1
                    break

            eval_rewards.append(eval_episode_reward)
        
        print("Evaluation score:", np.mean(eval_rewards))
        print("{} of {} episodes ended close to / at the final state.".format(episode_final, eval_runs))
        torch.save(agent.main_dqn.state_dict(), path+'/checkpoints/defi_{}_reward_{:.2f}.pth'.format(step_counter, np.mean(eval_rewards)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", default=3000000, type=int, help="Choose maximum amount of training steps")
    parser.add_argument("--start_learning", default=2000, type=int, help="Set amount of steps after which epsilon will be decreased and the agent will be learning")
    parser.add_argument("--replay_memory_size", default=10000, type=int, help="Set amount of past expiriences stored in the replay memory")
    parser.add_argument("--eps_annealing_steps", default=100000, type=int, help="Set amount of steps after which epsilon is decreased more slowly until max_steps")
    parser.add_argument("--agent_history_length", default=1, type=int, help="Choose how many past states are included in each input to update the agent")
    parser.add_argument("--evaluate_every", default=20000, type=int, help="Set evaluation interval")
    parser.add_argument("--eval_runs", default=10, type=int, help="Set amount of runs performed during evaluation")
    parser.add_argument("--network_update_every", default=1000, type=int, help="Set target network update frequency")
    parser.add_argument("--max_episode_length", default=550, type=int, help="Set maximum episode length")
    parser.add_argument("--batch_size", default=32, type=int, help="Set batch size retrieved from memory for learning")
    parser.add_argument("--learning_rate", default=0.0000625, type=float, help="Set learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="Set discount factor for Bellman equation")
    parser.add_argument("--eps_final", default=0.1, type=float, help="Set first value to which epsilon is lowered to after eps_annealing_steps")
    parser.add_argument("--eps_final_step", default=0.01, type=float, help="Set the second value to which epsilon is lowered to from eps_final until max_steps")
    
    parser.add_argument("--path", default=".", type=str, help="Set default saving path of logs and checkpoints")

    parser.add_argument("--resume_training", dest="resume", action='store_true', help="Load checkpoint from path folder and resume training")

    args = parser.parse_args()
    os.makedirs(args.path+'/checkpoints', exist_ok=True)
    os.makedirs(args.path+'/logs', exist_ok=True)

    model = None
    epsilon = 1.0
    step_counter = 0
    if args.resume:
        import glob
        paths = glob.glob(args.path+'/checkpoints/*.pt')
        p_cp = max(paths, key=os.path.getctime)
        model, step_counter, mean_reward, epsilon = load_model(p_cp)

        args.start_learning += step_counter
        args.max_steps += step_counter


    #print(args.replay_memory_size)
    train(args.path, step_counter=step_counter, eps=epsilon, model=model, max_steps=args.max_steps,start_learning=args.start_learning ,replay_memory_size=args.replay_memory_size, batch_size=args.batch_size, eps_annealing_steps=args.eps_annealing_steps, eps_final=args.eps_final, eps_final_step=args.eps_final_step, gamma=args.gamma, agent_history_length=args.agent_history_length, evaluate_every=args.evaluate_every, eval_runs=args.eval_runs, network_update_every=args.network_update_every, max_episode_length=args.max_episode_length, learning_rate=args.learning_rate)
    
        

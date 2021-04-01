import torch
import gym
import numpy as np
import argparse

import os, sys
sys.path.insert(0,'..')

from dfibert.tracker.nn.rl import Agent, Action_Scheduler

import dfibert.envs.tractography as RLTe


def train(path, max_steps=3000000, replay_memory_size=20000, eps_annealing_steps=100000, agent_history_length=1, evaluate_every=20000, eval_runs=5, network_update_every=10000, max_episode_length=200, learning_rate=0.0000625):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Init environment..")
    env = RLTe.EnvTractography(device = 'cpu')
    print("..done!")
    n_actions = env.action_space.n

    print("Init agent")
    state = env.reset()
    agent = Agent(n_actions=n_actions, inp_size=state.get_value().shape, device=device, hidden=512, agent_history_length=agent_history_length, memory_size=replay_memory_size, learning_rate=learning_rate)

    print("Init epsilon-greedy action scheduler")
    action_scheduler = Action_Scheduler(num_actions=n_actions, max_steps=max_steps, eps_annealing_steps=eps_annealing_steps, replay_memory_start_size=replay_memory_size, model=agent.main_dqn)

    step_counter = 0
        
    eps_rewards = []

    print("Start training...")
    while step_counter < max_steps:
        epoch_step = 0
        agent.main_dqn.train()
    ######## fill memory begins here
        while epoch_step < evaluate_every and epoch_step < replay_memory_size:
            state = env.reset()
            episode_reward_sum = 0
            terminal = False
            #fill replay memory while interacting with env
            #for episode_counter in range(max_episode_length):
            while not terminal:
                # get action with epsilon-greedy strategy       
                action = action_scheduler.get_action(step_counter, torch.FloatTensor(state.get_value()).to(device).unsqueeze(0))
                        
                next_state, reward, terminal = env.step(action)

                step_counter += 1
                epoch_step += 1

                # accumulate reward for current episode
                episode_reward_sum += reward


                agent.replay_memory.add_experience(action=action,
                                    state=state.get_value(),
                                    reward=reward,
                                    new_state=next_state.get_value(),
                                    terminal=terminal)


                state = next_state

            

                ####### optimization is happening here
                if step_counter > replay_memory_size:
                    loss = agent.optimize()


                ####### target network update
                if step_counter > replay_memory_size and step_counter % network_update_every == 0:
                    agent.target_dqn.load_state_dict(agent.main_dqn.state_dict())
                
                # if episode ended before maximum step
                if terminal:
                    terminal = False
                    state = env.reset()
                    break
                    
            eps_rewards.append(episode_reward_sum)
            
            if len(eps_rewards) % 100 == 0:
                with open(path+'/logs/rewards.dat', 'a') as reward_file:
                    print("[{}] {}, {}".format(len(eps_rewards), step_counter, np.mean(eps_rewards[-1000:])), file=reward_file)
                print("[{}] {}, {}, current eps {}".format(len(eps_rewards), step_counter, np.mean(eps_rewards[-1000:]), action_scheduler.eps_current) )
        torch.save(agent.main_dqn.state_dict(), path+'/checkpoints/fibre_agent_{}_reward_{:.2f}.pth'.format(step_counter, np.mean(eps_rewards[-1000:])))
    
    ########## evaluation starting here
        eval_rewards = []
        agent.main_dqn.eval()
        for _ in range(eval_runs):
            eval_steps = 0
            state = env.reset()
            eval_episode_reward = 0
            episode_final = 0
            while eval_steps < max_episode_length:
                action = action_scheduler.get_action(step_counter, torch.FloatTensor(state.get_value()).to(device).unsqueeze(0), evaluation=True)

                next_state, reward, terminal = env.step(action)

                eval_steps += 1
                eval_episode_reward += reward
                state = next_state

                if terminal:
                    terminal = False
                    if reward > 2.4:
                        episode_final += 1
                    break

            eval_rewards.append(eval_episode_reward)
        
        print("Evaluation score:", np.mean(eval_rewards))
        print("{} of {} episodes ended close to / at the final state.".format(episode_final, eval_runs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", default=3000000, type=int, help="Choose maximum amount of training steps")
    parser.add_argument("--replay_memory_size", default=10000, type=int, help="Set amount of past expiriences stored in the replay memory")
    parser.add_argument("--eps_annealing_steps", default=100000, type=int, help="Set amount of steps after which epsilon is decreased more slowly until max_steps")
    parser.add_argument("--agent_history_length", default=1, type=int, help="Choose how many past states are included in each input to update the agent")
    parser.add_argument("--evaluate_every", default=20000, type=int, help="Set evaluation interval")
    parser.add_argument("--eval_runs", default=5, type=int, help="Set amount of runs performed during evaluation")
    parser.add_argument("--network_update_every", default=10000, type=int, help="Set target network update frequency")
    parser.add_argument("--max_episode_length", default=200, type=int, help="Set maximum episode length")
    parser.add_argument("--learning_rate", default=0.0000625, type=float, help="Set learning rate")
    parser.add_argument("--path", default=".", type=str, help="Set default saving path of logs and checkpoints")

    args = parser.parse_args()
    os.makedirs(args.path+'/checkpoints', exist_ok=True)
    os.makedirs(args.path+'/logs', exist_ok=True)

    print(args.replay_memory_size)
    train(args.path, max_steps=args.max_steps, replay_memory_size=args.replay_memory_size, eps_annealing_steps=args.eps_annealing_steps, agent_history_length=args.agent_history_length, evaluate_every=args.evaluate_every, eval_runs=args.eval_runs, network_update_every=args.network_update_every, max_episode_length=args.max_episode_length, learning_rate=args.learning_rate)
    
        
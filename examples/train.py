import torch
import gym
import numpy as np
import argparse

import os, sys
sys.path.insert(0,'..')

from dfibert.tracker.nn.rl import Agent, Action_Scheduler

import dfibert.envs.RLtractEnvironment as RLTe


def get_best_action(state, env):
    best_actions = []
    rewards = []
    for i in range(env.action_space.n):
        next_state, reward,_ = env.step(i)
        rewards.append(float(reward))
        best_actions.append(reward)
        env.state = state
        env.stepCounter -= 1

    best_action= torch.argmax(torch.tensor(best_actions))
    return best_action, rewards[best_action]

def train(path, max_steps=3000000, batch_size=32, replay_memory_size=20000, start_learning=10000, eps_annealing_steps=100000, eps_final=0.1, eps_final_step=0.01, gamma=0.99, agent_history_length=1, evaluate_every=20000, eval_runs=5, network_update_every=10000, max_episode_length=200, learning_rate=0.0000625):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Init environment..")
    env = RLTe.RLtractEnvironment(stepWidth=0.3, action_space=20, device = 'cpu')
    print("..done!")
    n_actions = env.action_space.n

    print("Init agent")
    state = env.reset().getValue()
    agent = Agent(n_actions=n_actions, inp_size=state.shape, device=device, gamma=gamma, agent_history_length=agent_history_length, memory_size=replay_memory_size, batch_size=batch_size, learning_rate=learning_rate)

    print("Init epsilon-greedy action scheduler")
    action_scheduler = Action_Scheduler(num_actions=n_actions, max_steps=max_steps, eps_annealing_steps=eps_annealing_steps, replay_memory_start_size=start_learning, model=agent.main_dqn)

    step_counter = 0
        
    eps_rewards = []
    episode_lengths = []

    print("Start training...")
    while step_counter < max_steps:
        epoch_step = 0
        #agent.main_dqn.train()
    ######## fill memory begins here
        while (epoch_step < evaluate_every) or (step_counter < start_learning):
            state = env.reset()
            episode_reward_sum = 0
            terminal = False
            #fill replay memory while interacting with env
            #for episode_counter in range(max_episode_length):
            episode_step_counter = 0
            positive_run = 0
            while not terminal:
                # get action with epsilon-greedy strategy       
                action = action_scheduler.get_action(step_counter, torch.FloatTensor(state.getValue()).unsqueeze(0).to(device)) #influential_action=influential_action)
                next_state, reward, terminal = env.step(action)

                step_counter += 1
                epoch_step += 1
                episode_step_counter += 1

                _, optimal_reward = get_best_action(state, env)             # <- needs to be included in environment
                if reward < -0.05:                                          # <- needs to be included in environment
                    env.stepCounter -= 1                                    # <- needs to be included in environment                       

                reward = torch.tanh(1- (optimal_reward - reward))           # <- needs to be included in environment
                if reward >= 0.76:                                          # <- needs to be included in environment
                    reward = 1.                                             # <- needs to be included in environment
                elif reward < 0.1:                                          # <- needs to be included in environment
                    reward = -1.                                            # <- needs to be included in environment
                else:                                                       # <- needs to be included in environment
                    reward = 0.                                             # <- needs to be included in environment

                #if reward > 1.0:
                #    positive_run += 1

                # accumulate reward for current episode
                episode_reward_sum += reward


                agent.replay_memory.add_experience(action=action,
                                    state=state.getValue(),
                                    reward=reward,
                                    new_state=next_state.getValue(),
                                    terminal=terminal)


                state = next_state

            

                ####### optimization is happening here
                if step_counter > start_learning and step_counter % 4 == 0:
                    #if reward > 0.:
                    #    print("reward was positive: ", reward)
                    loss = agent.optimize()


                ####### target network update
                if step_counter > start_learning and step_counter % network_update_every == 0:
                    agent.target_dqn.load_state_dict(agent.main_dqn.state_dict())
                
                # if episode ended before maximum step
                if episode_step_counter >= max_episode_length:             # <- needs to be included in environment
                    terminal = True                         # <- needs to be included in environment
                if terminal:
                    terminal = False
                    state = env.reset()
                    episode_lengths.append(episode_step_counter)
                    break
                    
            eps_rewards.append(episode_reward_sum)
            
            if len(eps_rewards) % 1 == 0:
                with open(path+'/logs/rewards.dat', 'a') as reward_file:
                    print("[{}] {}, {}".format(len(eps_rewards), step_counter, np.mean(eps_rewards[-100:])), file=reward_file)
                print("[{}] {}, {}, current eps {}, avg steps {}".format(len(eps_rewards), step_counter, np.mean(eps_rewards[-100:]), action_scheduler.eps_current, np.mean(episode_lengths[-100:])) )
        #torch.save(agent.main_dqn.state_dict(), path+'/checkpoints/fibre_agent_{}_reward_{:.2f}.pth'.format(step_counter, np.mean(eps_rewards[-100:])))
    
    ########## evaluation starting here
        eval_rewards = []
        #agent.main_dqn.eval()
        for _ in range(eval_runs):
            eval_steps = 0
            state = env.reset()
            eval_episode_reward = 0
            episode_final = 0
            while eval_steps < max_episode_length:
                action = action_scheduler.get_action(step_counter, torch.FloatTensor(state.getValue()).unsqueeze(0).to(device), evaluation=True)
                next_state, reward, terminal = env.step(action)

                eval_steps += 1
                _, optimal_reward = get_best_action(state, env)             # <- needs to be included in environment
                if reward < -0.05:                                          # <- needs to be included in environment
                    env.stepCounter -= 1                                    # <- needs to be included in environment                       

                reward = torch.tanh(1- (optimal_reward - reward))           # <- needs to be included in environment
                if reward >= 0.76:                                          # <- needs to be included in environment
                    reward = 1.                                             # <- needs to be included in environment
                elif reward < 0.1:                                          # <- needs to be included in environment
                    reward = -1.                                            # <- needs to be included in environment
                else:                                                       # <- needs to be included in environment
                    reward = 0.                                             # <- needs to be included in environment

                eval_episode_reward += reward
                state = next_state

                if episode_step_counter >= 550:             # <- needs to be included in environment
                    terminal = True                         # <- needs to be included in environment
                if terminal:
                    terminal = False
                    if reward == 1.:
                        episode_final += 1
                    break

            eval_rewards.append(eval_episode_reward)
        
        print("Evaluation score:", np.mean(eval_rewards))
        print("{} of {} episodes ended close to / at the final state.".format(episode_final, eval_runs))
        # to do: add real checkpointing function
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

    args = parser.parse_args()
    os.makedirs(args.path+'/checkpoints', exist_ok=True)
    os.makedirs(args.path+'/logs', exist_ok=True)

    #print(args.replay_memory_size)
    train(args.path, max_steps=args.max_steps,start_learning=args.start_learning ,replay_memory_size=args.replay_memory_size, batch_size=args.batch_size, eps_annealing_steps=args.eps_annealing_steps, eps_final=args.eps_final, eps_final_step=args.eps_final_step, gamma=args.gamma, agent_history_length=args.agent_history_length, evaluate_every=args.evaluate_every, eval_runs=args.eval_runs, network_update_every=args.network_update_every, max_episode_length=args.max_episode_length, learning_rate=args.learning_rate)
    
        
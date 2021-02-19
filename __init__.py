from gym.envs.registeration import register

register(

id='RL_env-0', # id will be used in the main function to make an environment
entry_point='gym_FiberTracking.envs:RLenv', # environment class

)

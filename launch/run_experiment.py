import os, sys, gym, time, argparse
import numpy as np

from gym.envs.registration import register

sys.path.append('../exps')
from exp_setup_soccer import id_generator, run_experiment
print(sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) ))

#################################################################################################################

# SET EXPERIMENT PARAMETERS

game_ID = 'multigrid-soccer'
experiments  = 1
train_episodes = 0
total_episodes = 100          # Total episodes cannot be smaller than train episodies
step_cost = 0.01               # default: 0.001 animalai
transfer_frequency = 10        # values = [1,10,50] - default: 50
mutation_rate = 0.0          # values = [0.1,0.01,0.0] - default: 0.01
transfer_memories = True    # False: Normal SEC, True: Enable memory communication

view_screen = False             # To see simulation in real-time
file_path = '../saved_data/results/'


# SET MODEL PARAMETERS

agent = 'SEC'                 # MODELS = ['Reactive', 'SEC']
params = {}
params['prototype_length'] = 54          # length of the 1d observation vector

if agent == 'SEC':
    params['stm'] = 30                       # stms = [40,60,80,100]  - default: 15
    params['ltm'] = 5000                     # ltms = [25,50,100,500] - default: 500
    params['sequential_bias'] = True
    params['forgetting'] = 'FIFO'            # types = ['FIFO', 'SING', 'PROP'] - default: FIFO
    params['transfer_memories'] = transfer_memories
    params['transfer_cooldown'] = 25         # cooldowns = [10,15,20,25] - default: 25(1-2), 20(2-3), 15(3-4), 10(4-5) 
    params['transfer_type'] = 'PROP'         # types = ['BEST', 'RAND', 'PROP'] - default: PROP

#################################################################################################################


# RUN experiment

if __name__ == '__main__':
    try:
        for i in range(experiments):
            seed = i

            print('STARTING EXPERIMENT NUMBER '+str(i+1))
            print('MODEL: '+str(agent))
            print('ENV: '+str(game_ID))
            print('TRAINING EPISODES: '+str(train_episodes))
            print('TOTAL EPISODES: : '+str(total_episodes))

            total_rwd = run_experiment(
                model=agent, model_params=params, game=game_ID, seed_n=seed, filePath=file_path, 
                visualize=view_screen, train_eps=train_episodes, total_eps=total_episodes,
                step_cost=step_cost, transfer_frequency=transfer_frequency, mutation_rate=mutation_rate)

    except KeyboardInterrupt:
        print ('Simulation interrumpted!')

#################################################################################################################

import os, sys, torch, gym, time, string, argparse, pickle
import numpy as np
import random as rnd

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from gym.envs.registration import register

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='soccer', type=str)
args = parser.parse_args()

sys.path.append('../agents')

from agents import AgentReactive, AgentGridworld_SEC

from matplotlib import pylab as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#################################################################################################################


# HELPER FUNCTIONS

def id_generator(length=8, chars=string.ascii_lowercase + string.digits):
    return ''.join(rnd.choice(chars) for i in range(length))

#### RUNNING MEAN

def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y

#### PLOT LOSS

def plot_loss(save_dir, game_ID, exp_ID, i, losses):
    #with open(save_dir+game_ID+'_data_'+exp_ID+'.pickle', 'wb')
    plt.figure(figsize=(10,7))
    plt.plot(losses)
    plt.xlabel("Epochs",fontsize=22)
    plt.ylabel("Loss",fontsize=22)
    #plt.show()
    plt.savefig(save_dir+game_ID+'_'+exp_ID+'_agent'+str(i)+'.png')


#################################################################################################################


# EXPERIMENTAL SETUP FUNCTIONS

def run_experiment(model, model_params, game, seed_n, filePath, 
    visualize=True, train_eps=100, total_eps=100,
    step_cost=0.001, transfer_frequency=50, mutation_rate=0.001): 

    # ENVIRONMENT
    game_ID = game            # ENVS = ['Frostbite-v4', 'MsPacman-v4', 'Qbert-v4', 'Riverraid-v4', 'SpaceInvaders-v4']
    seed = seed_n
    render = visualize

    # EXPERIMENT
    model = model             # MODELS = ['Reactive', 'SEC', 'MFEC']
    params = model_params
    #agent_ID = 's'+str(seed)+'_'+model+'-'+id_generator(6)
    exp_ID = model+'_s'+str(seed)+id_generator(6)
    train_episodes = train_eps
    total_episodes = total_eps
    save_dir = filePath                 # Place to save the data

    step_cost = step_cost                               # default: 0.001 animalai
    transfer_frequency = transfer_frequency             # values = [10,50,100,500] - default: 50
    mutation_rate = mutation_rate
    prototype_length = params['prototype_length']       # default: 20


    if model == 'SEC':
        # FOR SEC
        stm = params['stm']                                 # stms = [40,60,80,100]  - default: 40
        ltm = params['ltm']                                 # ltms = [25,50,100,500] - default: 500
        sequential_bias = params['sequential_bias']
        forgetting_type = params['forgetting']
        transfer_memories = params['transfer_memories']
        transfer_cooldown = params['transfer_cooldown']
        transfer_type = params['transfer_type']

    # SET MODEL AND ENVIRONMENT

    # There already exists an environment generator that will make and wrap atari environments correctly.
    
    register(
        id='multigrid-soccer-v0',
        entry_point='gym_multigrid.envs:SoccerGame4HEnv10x15N2',
    )
    env = gym.make('multigrid-soccer-v0')

    #env = make_vec_env(game_ID+'-v0')
    #print("OBSERVATION SPACE: ", env.observation_space)
    #print("ACTION SPACE: ", env.action_space)
    #print("ACTIONS: ", env.action_space.n)

    _ = env.reset()
    #print("Observations", len(nObservations))

    nb_agents = len(env.agents)
    #print("Agent number: ", nb_agents)
    agents = []

    # generate X number of agents
    for n in range (nb_agents):
        if model == "Reactive":
            agent = AgentReactive(act_sp=env.action_space.n, env_n='multigrid-soccer-v0')
            #print('Testing Reactive Agents...')
        if model == "SEC":
            agent = AgentGridworld_SEC(
                act_sp=env.action_space.n, env_n='multigrid-soccer-v0', train_len=train_episodes,
                p_len=prototype_length, stm_len=stm, ltm_len=ltm, seq_bias=sequential_bias, frgt=forgetting_type,
                transf_cldwn=transfer_cooldown, transf_type=transfer_type)
            #print('Testing SEC Agents with STM_length '+str(stm)+' and LTM_length '+str(ltm))
            #print('Testing SEC Agents with Transfer Cooldown '+str(transfer_cooldown)+' and Transfer Type '+str(transfer_type))

        #agent.color = env_agent.index
        agents.append(agent)

    #################################################################################################################


    # SET EPISODES VARIABLES
    current_episodes = 0    # Current number of episodes played
    eps_length = 0          # Length in steps of the current episode
    frames = 0
    transfer_active = False
    
    rewards = []            # Rewards obtained in the current timestep
    eps_rewards = np.array([0., 0., 0., 0.])    # Accumulated rewards obtained in the current episode
    eps_layers = []
    total_rwd = 0

    # SET LOG VARIABLES
    data = []
    log_rwd, log_eps, log_layers = [], [], []
    log_mem_sent, log_mem_rec, log_mems = [], [], []

    #################################################################################################################

    # BEGIN MODEL LEARNING

    #nObservations, nRewards, nDone, nInfo = env.reset() 
    action = [env.action_space.sample() for _ in range(nb_agents)]
    nObservations, _, nDone, _ = env.step(action)
    #print("Observations", len(nObservations))
    #flat_obs = nObservations[0].flatten()
    #print("Observations flatten", len(flat_obs))
    nStates1 = []
    nLosses = []
    #print("Observations", len(nObservations))

    while current_episodes < total_episodes:
        frames +=1
        eps_length += 1
        #print("STEP ", eps_length)

        if render: 
            env.render(mode='human', highlight=True)
            # highlight - visualizes the range of vision of the agents
        else:
            env.render(mode='rgb_array', highlight=False)
        
        nActions, nLayers = [], [[],[],[],[]]

        for i, agent in enumerate(agents):

            # TRANSFER MEMORIES 
            if model == 'SEC': 
                # 0. check that is the right round to transfer memories
                if transfer_memories == True and transfer_active == True:
                    # 1. check if agent is able to transfer memories
                    if agent.transfer_ready == True:
                        #print('Agent {} is ready for transfer'.format(i))
                        # 2. check for other agents in visual range
                        flat_obs = nObservations[i].flatten()
                        detected_agents_idx = agent.detect_agents(flat_obs)
                        # 3. transfer memories between agents
                        if len(detected_agents_idx) > 0:
                            #print('Agent {} is sending a memory'.format(i))
                            memory = agent.retrieve_memory()
                            #agents[k].receive_memory(memory) for j,k in detected_agents_idx
                            for k in detected_agents_idx:
                                #if agents[k].transfer_ready == True:
                                #print('Agent {} sent a memory'.format(i))
                                agent.memories_sent += 1
                                agents[k].receive_memory(memory, mutation_rate, save_dir, exp_ID) 
                                #print('Agent {} received a memory'.format(k))


            # GET NEW OBSERVATIONS AND TAKE ACTION
            #print("Agent {} observation {}".format(i+1, nObservations[i]))
            flat_obs = nObservations[i].flatten()
            nActions.append(agent.step(flat_obs))

            if model == 'SEC': nLayers[i].append(agent.layer_chosen)


        #print("nActions", nActions)
        nObservations, nRewards, nDone, nInfo = env.step(nActions)
        #print("nRewards: ", type(nRewards))
        #print("nObservations: ", nObservations)

        nRewards[nRewards > 0] -= step_cost * eps_length

        if np.sum(nRewards) > 0: 
            rewards.append(nRewards.tolist())  
            #print("nRewards: ", nRewards)
            #print("rewards: ", rewards)           
            eps_rewards = np.sum(rewards, axis=0)
            #print("eps_rewards: ", eps_rewards)
       
        nRewards[nRewards <= 0] -= step_cost
        #print("nRewards: ", type(nRewards))
        #print("nRewards: ", nRewards)

        for i, agent in enumerate(agents):
            if model == 'SEC': 
                agent.update_STM()
                agent.update_LTM(nRewards[i])

        nStates1 = []  
        #state1 = state2

        #if (eps_length%100) == 0:
            #print("STEP ", eps_length)

        if nDone:
            #print("Episode DONE!!")
            for i, agent in enumerate(agents):
                if model == 'SEC': agent.reset_STM()
                    #if agent.epsilon > 0.1: #R
                        #epsilon -= (1/total_episodes)

            ### LOG EPISODE DATA

            log_eps.append(eps_length)
            log_rwd.append(eps_rewards)

            total_rwd = np.sum(np.array(log_rwd), axis=0)
            #print("total_rwd: ", total_rwd)

            if model == 'SEC':
                for i, layers in enumerate (nLayers):
                    eps_layers.append(round(layers.count('C') / len(layers),2))

            log_layers.append(eps_layers)

            if model == 'SEC': 
                mems, mems_sent, mems_rec = [],[],[]
                for i, agent in enumerate(agents):
                    mems.append(agent.acquired_memories)
                    mems_sent.append(agent.memories_sent)
                    mems_rec.append(agent.memories_received)

            log_mems.append(mems)
            log_mem_sent.append(mems_sent)
            log_mem_rec.append(mems_rec)

            ### RESET VARIABLES FOR NEW EPISODE

            rewards = []
            eps_length = 0
            eps_layers = []
            eps_rewards = np.array([0., 0., 0., 0.])

            _ = env.reset()
            #nObservations, nRewards, nDone, nInfo = env.reset() 
            action = [env.action_space.sample() for _ in range(nb_agents)]
            nObservations, _, nDone, _ = env.step(action)
            #print("Observations", len(nObservations))

            current_episodes += 1
            if (current_episodes%500) == 0:
                print("EPISODE ", current_episodes)

            if current_episodes >= train_episodes: # during train episodes, agents do not share memories
                # Set transfer memory ratio --- DEFAULT VALUE 10: Perform memory transfer every 10 episodes
                if current_episodes%transfer_frequency == 0: 
                    transfer_active = True
                    #print('TRANSFER ACTIVE')
                else: 
                    transfer_active = False
            #break


    #################################################################################################################

    # SAVE DATA
    full_memories = {}
    if model == 'SEC':
        #print("Saving memories...")
        for i, agent in enumerate(agents):
            #agent_memories = np.array(agent.CL.LTM)
            agent_memories = agent.CL.LTM
            full_memories['agent_'+str(i)] = agent_memories

    exp_params = {}
    exp_params['model'] = model 
    exp_params['env'] = game_ID 
    exp_params['total_episodes'] = total_episodes                     
    exp_params['train_episodes'] = train_episodes                    
    exp_params['step_cost'] = step_cost    
    exp_params['transfer_frequency'] = transfer_frequency
    exp_params['mutation_rate'] = mutation_rate         
 

    data_logs = {
      "exp_params": exp_params,
      "model_params": params,
      "rewards": log_rwd,
      "episode_len": log_eps,
      "layer_activity": log_layers,
      "loss": nLosses,
      "memories": log_mems,
      "memories_sent": log_mem_sent,
      "memories_received": log_mem_rec,
      "memories_list": full_memories
    }


    print("Saving data...")
    with open(save_dir+game_ID+'_'+exp_ID+'.pickle', 'wb') as f:
        pickle.dump(data_logs, f)

    env.close()

    print("SIMULATION COMPLETE")

    return total_rwd

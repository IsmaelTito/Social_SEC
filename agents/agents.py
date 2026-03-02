import numpy as np

import sys
sys.path.append('../layers/')

from reactive_layer import ReactiveLayer_Double
from contextual_layer_SEC import ContextualLayer_SEC


class AgentReactive(object):

    def __init__(self, act_sp, env_n):
        self.RL = ReactiveLayer_Double(action_space=act_sp, env_name=env_n)

        self.env_name = env_n
        self.action_space = act_sp      # can be a list "[3, 3]" or a integer "6"

    def reset(self, t=250):
        pass

    def step(self, state):
        # ACTION SELECTION PHASE

        # Action proposed by the Reactive Layer.
        action_RL = self.RL.action_selection()

        return action_RL


class AgentGridworld_SEC(AgentReactive):

    def __init__(self, act_sp, env_n, train_len=1000, p_len=20, stm_len=50, ltm_len=500, seq_bias=True, frgt='FIFO', load_ltm=False,
        transf_cldwn=25, transf_type='PROP'):
        super().__init__(act_sp, env_n)
        self.CL = ContextualLayer_SEC(action_space=act_sp, pl=p_len, stm=stm_len, ltm=ltm_len,
            forget=frgt, sequential_bias=seq_bias, load_ltm=load_ltm)

        self.previous_couplet = np.array([np.zeros(p_len), np.random.choice(self.action_space)], dtype=object)

        self.steps = 0
        self.minimum_steps = train_len

        self.layer_chosen = 'R'

        self.minimum_memory = 1
        self.acquired_memories = 0
        self.memories_sent = 0
        self.memories_received = 0

        self.transfer_cooldown = transf_cldwn # timesteps waiting between memory transfers: test with 25(1-2), 20(2-3), 15(3-4), 10(4-5)
        self.transfer_count = 0
        self.transfer_ready = False
        self.transfer_type = transf_type # can be 'BEST', 'RAND', 'PROP'
        #self.color = 0

    def step(self, state):
        self.steps += 1
        self.acquired_memories = len(self.CL.LTM[2])

        # UPDATE MEMORY TRANSFER DYNAMICS
        if self.transfer_ready == False and self.acquired_memories >= self.minimum_memory:
            self.transfer_count += 1
            #cooldown = self.transfer_cooldown - self.transfer_count
            #print("AGENT MEMORY TRANSFER COOLDOWN... ", cooldown)

        if self.transfer_count == self.transfer_cooldown:
            #print("AGENT READY FOR MEMORY TRANSFER")
            self.transfer_ready = True
            self.transfer_count = 0


        # ACTION SELECTION PHASE
        action = -1 # np.array([0, 0])
        # Action proposed by the Reactive Layer.
        action_RL = self.RL.action_selection()
        # Action proposed by the Contextual Layer.
        action_CL = self.CL.action_selection(state)

        # Arbitration procedure for final action selection (CL > RL).

        if self.env_name == 'DoubleTMaze':
            # For AnimalAI games: Chose CL action if the reconstruction error (ie. discrepancy) is smaller than the threshold value AND if there is content in the CL's memory
            if self.rec_error <= self.rec_threshold and self.acquired_memories > self.minimum_memory:
                #ac_indx = int(action_CL[0]*self.action_space[0] + action_CL[1])    # convert action[x,x] into an integer y for correct output in atari
                action = action_CL
                self.layer_chosen = 'C' #Reactive Layer default selection
                #print ('Contextual ', action)
            else:
                action = action_RL
                self.layer_chosen = 'R' #Reactive Layer default selection
                #print ('Reactive ', action)
        elif self.env_name == 'multigrid-soccer-v0':
            # For AnimalAI games: Chose CL action if the reconstruction error (ie. discrepancy) is smaller than the threshold value AND if there is content in the CL's memory
            if self.acquired_memories > self.minimum_memory:
                #ac_indx = int(action_CL[0]*self.action_space[0] + action_CL[1])    # convert action[x,x] into an integer y for correct output in atari
                action = action_CL
                self.layer_chosen = 'C' #Reactive Layer default selection
                #print ('Contextual ', action)
            else:
                action = action_RL
                self.layer_chosen = 'R' #Reactive Layer default selection
                #print ('Reactive ', action)
        else:
            # For Atari games: Chose CL action after a minimum number of exploration steps have been taken
            if self.steps > self.minimum_steps:
                action = action_CL
                self.layer_chosen = 'C' #Reactive Layer default selection
                #print ('Contextual ', action)
            else:
                action = action_RL
                self.layer_chosen = 'R' #Reactive Layer default selection
                #print ('Reactive ', action)

        # Store couplet for next update of STM and LTM based on next reward.
        #ac = [int(action/self.action_space[0]), int(action%self.action_space[1])]   #convert action from integer y to [x,x] for keeping internal structure
        self.previous_couplet = [state, action]
        self.update_STM()

        return action

    def update_STM(self):
        # MEMORY UPDATE PHASE 1
        # Update STM based on previous (state,action) couplet
        self.CL.update_STM(self.previous_couplet)
        self.CL.update_sequential_bias()

    def update_LTM(self, reward):
        # MEMORY UPDATE PHASE 2
        # Update LTM based on current reward
        self.CL.update_LTM(reward)

    def reset_STM(self):
        # MEMORY RESET when finishing an episode
        self.CL.reset_STM()
        self.CL.reset_sequential_bias()

    def count_memories(self):
        # MEMORY COUNT in single units (Stored LTMs x STM length)
        return self.CL.get_LTM_length()

    def detect_agents(self, obs):
        # check for other agents in visual range
        x = [i+1 for i, j in enumerate(obs) if j == 10]

        agent_idx = []

        # identify identity of the detected agents
        for i in range (len(x)):
            # check that detected agent is not self
            if obs[x[i]+4] == 0:
                #print("agent idx: ", obs[x[i]]-1)
                agent_idx.append(obs[x[i]]-1)

        return agent_idx

    def retrieve_memory(self):
        # GET THE BEST MEMORY OF THE AGENT
        memory = self.CL.extract_memory(self.transfer_type)
        #self.memories_sent += 1
        self.transfer_ready = False
        return memory

    def receive_memory(self, memory, mutation_rate, save_dir, exp_ID):
        # RECEIVE MEMORY FROM OTHER AGENT
        self.CL.inject_memory(memory, mutation_rate, save_dir, exp_ID)
        self.memories_received += 1
        #self.transfer_count = 0
        #self.transfer_ready = False

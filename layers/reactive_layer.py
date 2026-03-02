import numpy as np
from numpy import random


'''
    Parent Reactive Layer class implementing a Randon Action Selection algorithm based on possible env actions
'''

class ReactiveLayer(object):

    def __init__(self, action_space):
        self.env_action_space = action_space
        #print('self.env_action_space :', self.env_action_space)
        #self.action = 0 if type(self.env_action_space == int) else np.array([-1, -1])       # can be a list "[0, 0]...[1, 2]" or a integer "0...5"  
        if type(self.env_action_space) != list:
            self.action = 0 
            #print("type not list")
        else: 
            #print("type list")
            self.action = np.array([-1, -1]) 

    def action_selection(self):
        #action = self.env_action_space.sample()
        self.action = np.random.choice(self.env_action_space)
        return self.action



'''
    Child Reactive Layer class implementing a Pseudo-Random Walk for AnimalAI-v1
'''

class ReactiveLayer_RW(ReactiveLayer):

    def __init__(self, action_space):
        super().__init__(action_space) 
         # Variables related to random exploration
        self.random_straight_range = 20
        self.random_turn_range = 20
        self.random_straight = random.randint(1, self.random_straight_range)
        self.random_turn = random.randint(1, self.random_turn_range)
        self.random_action = 0
        self.random_movement_counter = 0
        self.converted_list_space = [3, 3]

    def action_selection(self):    
        self.action = self.random_walk()
        return self.action

    def random_walk(self):
        step = 0

        """Generate a series of straight moves followed by a series of turning moves in a random direction"""
        self.random_movement_counter += 1

        # check if series of random actions has been completed and if so reset variables
        if self.random_movement_counter > self.random_turn + self.random_straight:
            # reset variables
            self.random_movement_counter = 0
            self.random_action = random.randint(0, 2)
            self.random_straight = random.randint(1, self.random_straight_range+1)
            self.random_turn = random.randint(1, self.random_turn_range+1)

        # take a series of straight movements
        if self.random_movement_counter <= self.random_straight:
            step = [1, 0]

        # take a series of rotation movements
        elif self.random_straight < self.random_movement_counter <= self.random_turn + self.random_straight:
            if self.random_action == 0:
                step = [0, 1]
            else:
                step = [0, 2]

        if type(self.env_action_space) != list:
            step = int(step[0]*self.converted_list_space[0] + step[1])


        return step


class ReactiveLayer_Double(ReactiveLayer_RW):
    
    def __init__(self, action_space, env_name):
        super().__init__(action_space)
        self.env_name = env_name

    def action_selection(self):
        if self.env_name == 'DoubleTMaze':
            self.action = self.random_walk()
        else:
            self.action = np.random.choice(self.env_action_space)
        
        return self.action

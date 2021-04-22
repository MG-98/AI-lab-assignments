#!/usr/bin/env python
# coding: utf-8

# In[149]:


from collections import deque
import numpy as np
import math
from heapq import heappush, heappop
from itertools import count


# In[150]:


class Problem:
    '''
    Abstract base class for problem formulation.
    It declares the expected methods to be used by a search algorithm.
    All the methods declared are just placeholders that throw errors if not overriden by child "concrete" classes!
    '''

    def __init__(self):
        '''Constructor that initializes the problem. Typically used to setup the initial state and, if applicable, the goal state.'''
        self.init_state = None

    def actions(self, state):
        '''Returns an iterable with the applicable actions to the given state.'''
        raise NotImplementedError

    def result(self, state, action):
        '''Returns the resulting state from applying the given action to the given state.'''
        raise NotImplementedError

    def goal_test(self, state):
        '''Returns whether or not the given state is a goal state.'''
        raise NotImplementedError

    def step_cost(self, state, action):
        '''Returns the step cost of applying the given action to the given state.'''
        raise NotImplementedError
    
    def heuristic(self, state):
        '''Returns the heuristic value of the given state, i.e., the estimated number of steps to the nearest goal state.'''
        raise NotImplementedError

class Node:
    '''Node data structure for search space bookkeeping.'''

    def __init__(self, state, parent, action, path_cost, heuristic):
        '''Constructor for the node state with the required parameters.'''
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.g = path_cost
        self.h = heuristic
        self.f = path_cost + heuristic

    @classmethod
    def root(cls, init_state):
        '''Factory method to create the root node.'''
        init_state = problem.init_state
        return cls(init_state, None, None, 0, problem.heuristic(init_state))

    @classmethod
    def child(cls, problem, parent, action):
        '''Factory method to create a child node.'''
        child_state = problem.result(parent.state, action)
        return cls(
            problem.result(parent.state, action),
            parent,
            action,
            parent.g + problem.step_cost(parent.state, action),
            problem.heuristic(child_state))
    
def solution(node):
    '''A method to extract the sequence of actions representing the solution from the goal node.'''
    actions = []
    cost = node.g
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    actions.reverse()
    return actions, cost


# In[151]:



counter = count()

def greedy_best_first(problem, verbose=False):
    '''Greedy best-first search implementation.'''
    frontier = [(None, None, Node.root(problem))]
    explored = set()
    #if verbose: visualizer = Visualizer(problem)
    while frontier:
     #   if verbose: visualizer.visualize(frontier)
        _, _, node = heappop(frontier)
        if node.state in explored: continue
        if problem.goal_test(node.state):
            return solution(node)
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = Node.child(problem, node, action)
            if child.state not in explored:
                heappush(frontier, (child.h, next(counter), child))


# In[152]:


#################### PART A #############################


# In[153]:



counter = count()

def A_star(problem, verbose=False):
    '''Greedy best-first search implementation.'''
    frontier = [(None, None, Node.root(problem))]
    explored = set()
    #if verbose: visualizer = Visualizer(problem)
    while frontier:
     #   if verbose: visualizer.visualize(frontier)
        _, _, node = heappop(frontier)
        if node.state in explored: continue
        if problem.goal_test(node.state):
            return solution(node)
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = Node.child(problem, node, action)
            if child.state not in explored:
                heappush(frontier, (child.f, next(counter), child))
                


counter = count()

def uniform_cost_search(problem, verbose=False):
    '''Greedy best-first search implementation.'''
    frontier = [(None, None, Node.root(problem))]
    explored = set()
    #if verbose: visualizer = Visualizer(problem)
    while frontier:
     #   if verbose: visualizer.visualize(frontier)
        _, _, node = heappop(frontier)
        if node.state in explored: continue
        if problem.goal_test(node.state):
            return solution(node)
        explored.add(node.state)
        for action in problem.actions(node.state):
            child = Node.child(problem, node, action)
            if child.state not in explored:
                heappush(frontier, (child.g, next(counter), child))


# In[154]:


############################ PART B ###################################


# In[187]:


# formation of the problem
class Maze(Problem):
    '''3x3 Sliding Puzzle problem formulation.'''

    def __init__(self, x1, init_state, goal_state):
        # assert init_state.count('*') == (1,1)
        # assert goal_state.count('*') == (1,1)
        self.init_state = init_state
        self._goal_state = goal_state
        self._m = x1
        self._action_values = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def actions(self, state):
        # up
        if state[0] - 1 > 0:
            if self._m[(state[0] - 1, state[1])] != '#':
                # update # swap
                yield 'up'
        # down
        if state[0] + 1 < 5:
            if self._m[(state[0] + 1, state[1])] != '#':
                yield 'down'
        # left
        if state[1] - 1 > 0:
            if self._m[(state[0], state[1] - 1)] != '#':
                yield 'left'
        # right
        if state[0] - 1 < 21:
            if self._m[(state[0], state[1] + 1)] != '#':
                yield 'right'

    def result(self, state, action):
        def swap(i, j):
            '''Auxiliary function for swapping two elements in a tuple.'''
            self._m[i], self._m[j] = self._m[j], self._m[i]
            state = j
            return state

        return swap(state, (state[0] + self._action_values[action][0], state[1] + self._action_values[action][1]))

    def goal_test(self, state):
        return state == self._goal_state
    
    def heuristic(self, state):
        #return math.sqrt(sum([(a - b) ** 2 for a, b in zip(state, self._goal_state)]))
        return sum(x != y for x, y in zip(state, self._goal_state))

    def step_cost(self, state, action):
        return 1


# In[188]:


# Preprocessing
maze =  '''
   #                #
   #                #
          ####      #
  ##      #     #####
  *#      #      +   
'''

def maze_preprocessing(maze):
    maze = maze.replace(' ','0')
    mazee = list(maze)
    mazee.remove('\n')
    x = np.reshape( mazee , (5,22) )
    x = x [x != '\n']
    x1 = np.reshape( x , (5,21) )
    return x1

def goal_index(x1):
    z = np.where(x1 == '+')
    z1  = list(z[0])[0]
    z2 =  list(z[1])[0]
    goal= (z1 ,z2)
    return goal

def init_index(x1):
    z = np.where(x1 == '*')
    z1  = list(z[0])[0]
    z2 =  list(z[1])[0]
    init= (z1 ,z2)
    return init

p = maze_preprocessing(maze)
goal = goal_index(p)
init = init_index(p)

p1 = Maze( p , init , goal)
    
    


# In[189]:


# ADIMICABILITY in the hurestic function grantee optimality in tree
# Consesitency in the hurestic grantee optimality in graphs


# In[195]:


get_ipython().run_line_magic('prun', 'print(uniform_cost_search(p1))')


# In[194]:


get_ipython().run_line_magic('prun', 'print(A_star(p1))')


# In[192]:


get_ipython().run_line_magic('prun', 'print(greedy_best_first(p1))')


# In[ ]:


#Comparing between the algorithms :

#We will compare between the algorithms in two deimensions -if we can call it that :)- the time and the performance when we change the hueristic funstion :

# 1- Eculadian distance :

# A* : time :0.002         / #of swaps :    286             / #of push: 74
# uniform search : time :  0.006       / #of swaps : 334                / #of push:85
# greedy best first : time : 0.003        / #of swaps : 140                / #of push: 71

#2- Manhattan distance

# A* : time :0.009         / #of swaps :    159             / #of push:82
# uniform search : time : 0.020       / #of swaps : 334                / #of push:85
# greedy best first : time :  .009       / #of swaps :  131               / #of push:70


# In[ ]:





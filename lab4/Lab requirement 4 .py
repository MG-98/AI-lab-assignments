#!/usr/bin/env python
# coding: utf-8

# In[65]:


class Game:
    '''
    Abstract game class for game formulation.
    It declares the expected methods to be used by an adversarial search algorithm.
    All the methods declared are just placeholders that throw errors if not overriden by child "concrete" classes!
    '''
    
    def __init__(self):
        '''Constructor that initializes the game. Typically used to setup the initial state, number of players and, if applicable, the terminal states and their utilities.'''
        self.init_state = None
    
    def player(self, state):
        '''Returns the player whose turn it is.'''
        raise NotImplementedError
    
    def actions(self, state):
        '''Returns an iterable with the applicable actions to the given state.'''
        raise NotImplementedError
    
    def result(self, state, action):
        '''Returns the resulting state from applying the given action to the given state.'''
        raise NotImplementedError
    
    def terminal_test(self, state):
        '''Returns whether or not the given state is a terminal state.'''
        raise NotImplementedError
    
    def utility(self, state, player):
        '''Returns the utility of the given state for the given player, if possible (usually, it has to be a terminal state).'''
        raise NotImplementedError


# In[210]:


from enum import Enum

class TicTacToe(Game):
    '''Tic-tac-toe game formulation.'''
    
    class Players(Enum):
        '''Enum with the players in tic-tac-toe.'''
        X = 'X'
        O = 'O'
    
    def _won(self, state, player):
        '''Auxiliary function for checking if a player has won.'''
        return any(all(state[0][i][j] is player for i in range(3)) for j in range(3))             or any(all(state[0][j][i] is player for i in range(3)) for j in range(3))             or all(state[0][i][i] is player for i in range(3))
    
    def __init__(self):
        self.init_state = ((None,) * 3,) * 3, None
    
    def player(self, state):
        return TicTacToe.Players.O if state[1] else TicTacToe.Players.X
    
    def actions(self, state):
        return ((i, j) for i, row in enumerate(state[0]) for j, player in enumerate(row) if not player)
    
    def result(self, state, action):
        mutable_grid = list(state[0])
        mutable_row = list(mutable_grid[action[0]])
        mutable_row[action[1]] = self.player(state)
        mutable_grid[action[0]] = tuple(mutable_row)
        return tuple(mutable_grid), not state[1]
    
    def terminal_test(self, state):
        return all(state[0][i][j] is not None for i in range(3) for j in range(3)) or any(self._won(state, player) for player in TicTacToe.Players)
    
    def utility(self, state, player):
        for p in TicTacToe.Players:
            if self._won(state, p):
                return 1 if p is player else -1
        return 0
    def evaluation(self, state, player):
        def On(state):
            count = []
            d = []
            for i in range(3):
                d = d + [state[0][i][i]]
                x = []
                count = count  + [list(state[0][i]).count(game.Players.O)]
                for j in range(3):
                    x = x + [state[0][j][i]]
                count  = count + [x.count(game.Players.O)]
            count = count + [d.count(game.Players.O)]
            o1 = count.count(1)
            o2 = count.count(2)
            return 3*o2 + o1
        def Xn(state):
            count = []
            d = []
            for i in range(3):
                d = d + [state[0][i][i]]
                x = []
                count = count  + [list(state[0][i]).count(game.Players.X)]
                for j in range(3):
                    x = x + [state[0][j][i]]
                count  = count + [x.count(game.Players.X)]
            count = count + [d.count(game.Players.X)]
            x1 = count.count(1)
            x2 = count.count(2)
            return 3*x2 + x1
        if self.terminal_test(state) : return self.utility(state, player)
        else:
            if player == 'O': return On(state)
            else : return Xn(state)
            


# In[214]:


from math import inf

def h_minimax(game, state , depth):
    '''Minimax implementation.'''
    player = game.player(state)
    d = 0
    def cut_off(state , d):
        if game.terminal_test(state): return True
        if d > depth:return True
        return False
    def max_value(state,d):
        if cut_off(state,d): 
            d = 0 
            return game.evaluation(state, player)
        maxi = -inf
        d = d+1
        for action in game.actions(state):
            maxi = max(maxi, min_value(game.result(state, action), d))
        return maxi
    def min_value(state , d):
        if cut_off(state,d): 
            d =0
            return game.evaluation(state, player)
        
        mini = +inf
        d=d+1
        for action in game.actions(state):
            mini = min(mini, max_value(game.result(state, action),d))
        return mini
    return max(((min_value(game.result(state, action),d), action) for action in game.actions(state)), key=lambda entry: entry[0])[1]


# In[215]:


game = TicTacToe()


# In[217]:


state = game.init_state
d=3
while(not game.terminal_test(state)):
    action = h_minimax(game, state , 4 )
    print(action)
    state = game.result(state, action)
    print("---------------------")
    print(state)
state


# In[ ]:





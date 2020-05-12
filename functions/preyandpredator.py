# This file is to realize the game environment

import random

# action space:
MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3
MOVE_STAY  = 4

class Game():

    def __init__(self, xlen, ylen):

        # the grid world attr
        self.envxlen = xlen
        self.envylen = ylen

        # position of predator and prey
        self.x_f1, self.y_f1 = 0, 0
        self.x_f2, self.y_f2 = 0, 0

        self.x_g1, self.y_g1 = 0, 0
        self.x_g2, self.y_g2 = 0, 0

        # init the env
        self.reset()

    def get_state(self):

        state = [self.x_f1, self.y_f1, self.x_f2, self.y_f2,
                 self.x_g1, self.y_g1, self.x_g2, self.y_g2]

        return state

    def display(self):
        print('f1:[{0},{1}], f2:[{2},{3}], g1:[{4},{5}], g2:[{6},{7}]'.format(
            self.x_f1,self.y_f1,self.x_f2,self.y_f2,self.x_g1,self.y_g1,self.x_g2,self.y_g2))

        return

    def get_unit_pos(self, unit):

        if unit is None:
            return

        x, y = 0, 0

        if unit is 'f1':
            x = self.x_f1
            y = self.y_f1
        if unit is 'f2':
            x = self.x_f2
            y = self.y_f2
        if unit is 'g1':
            x = self.x_g1
            y = self.y_g1
        if unit is 'g2':
            x = self.x_g2
            y = self.y_g2

        return x, y

    def set_unit_pos(self, unit, x, y):
        if unit is None:
            return

        if unit is 'f1':
            self.x_f1 = x
            self.y_f1 = y
        if unit is 'f2':
            self.x_f2 = x
            self.y_f2 = y
        if unit is 'g1':
            self.x_g1 = x
            self.y_g1 = y
        if unit is 'g2':
            self.x_g2 = x
            self.y_g2 = y

    def move(self, unit, action):
        x, y = self.get_unit_pos(unit)

        if   action == MOVE_UP:
            y = (y-1 if y>0 else 0)
        elif action == MOVE_DOWN:
            y = (y+1 if y<self.envylen-1 else y)
        elif action == MOVE_LEFT:
            x = (x-1 if x>0 else 0)
        elif action == MOVE_RIGHT:
            x = (x+1 if x<self.envxlen-1 else x)
        elif action == MOVE_STAY:
            pass
        else:
            print('input wrong action')
            return

        self.set_unit_pos(unit, x, y)

    def step(self, fa1, fa2, ga1, ga2):

        self.move('f1', fa1)
        self.move('f2', fa2)
        self.move('g1', ga1)
        self.move('g2', ga2)

        state = self.get_state()

        # reward: r2 for predator while r1 for prey
        r2 = -self.get_Distance()
        r1 = -r2

        return state, r1, r2

    def get_Distance(self):

        D = max(min(abs(self.x_f1-self.x_g1)+abs(self.y_f1-self.y_g1),
                    abs(self.x_f2-self.x_g1)+abs(self.y_f2-self.y_g1)),
                min(abs(self.x_f1-self.x_g2)+abs(self.y_f1-self.y_g2),
                    abs(self.x_f2-self.x_g2)+abs(self.y_f2-self.y_g2)))

        return D

    def reset(self):
        self.x_f1 = random.randint(0, self.envxlen - 1)
        self.x_f2 = random.randint(0, self.envxlen - 1)
        self.x_g1 = random.randint(0, self.envxlen - 1)
        self.x_g2 = random.randint(0, self.envxlen - 1)
        self.y_f1 = random.randint(0, self.envylen - 1)
        self.y_f2 = random.randint(0, self.envylen - 1)
        self.y_g1 = random.randint(0, self.envylen - 1)
        self.y_g2 = random.randint(0, self.envylen - 1)

        return self.get_state()

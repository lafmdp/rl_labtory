'''
  Get f(s,a)->s'
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/12/10
'''

import gym
import numpy as np

class self_mujuco():

    def __init__(self, id):

        self.env = gym.make(id)
        self.qpos_len = len(self.env.sim.data.qpos)
        self.qvel_len = len(self.env.sim.data.qvel)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # print("qpos_len:{},qvel_len:{},action_space:{},obs_space:{}".format(
        #     self.qpos_len,self.qvel_len, self.action_space, self.observation_space
        # ))
        self.action_length = self.action_space.shape[0]
        self.observation_length = self.observation_space.shape[0]
        self.env.reset()

    def forward(self, state, action):
        '''
        Given s,a, get s'.
        '''

        self.env.sim.set_state(state)
        s_, r, done, _ = self.env.step(action)
        self.env.reset()

        return s_, r, done, _

    def get_state(self):
        return self.env.state_vector()

    def get_sim_state(self):
        return self.env.sim.get_state()

    def set_sim_state(self, state):
        self.env.reset()
        self.env.sim.set_state(state)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

if __name__ == "__main__":

    test_mujuco = self_mujuco('Walker2d-v2')
    a = np.random.normal(0, 0.5, [test_mujuco.observation_length])

    s = test_mujuco.env.reset()
    #print('reset s:',s)
    print('set state:', a)
    print('real_state:',test_mujuco.get_sim_state())
    print("sim state:", test_mujuco.get_sim_state())
    #print('qpos',test_mujuco.env.sim.data.qpos)
    #print('qvel', test_mujuco.env.sim.data.qvel)
    s_ = test_mujuco.forward(a, np.random.random([test_mujuco.action_length]).tolist())

    print('next_state:', s_)
    print('real_next_state:', test_mujuco.env.state_vector())

'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/1/21
'''

import gym
from utils.utils_new import *


class action_noise_env():

    def __init__(self, noise, name):

        self.noise = noise
        self.env = gym.make(name)

    def reset(self):
        s = self.env.reset()

        return s


    def render(self):
        self.env.render()


    def step(self, a):
        a = np.array(a)
        a = a + np.random.normal(0,self.noise,a.shape)
        a = np.clip(a, -1, 1)

        return self.env.step(a)


    def close(self):
        self.env.close()


def get_new_gravity_env(variety, env_name):

    update_target_env_gravity(variety, env_name)
    env = gym.make(env_name)

    return env


def get_source_env(env_name="Walker2d-v2"):

    update_source_env(env_name)
    env = gym.make(env_name)

    return env


def get_new_density_env(variety, env_name):

    update_target_env_density(variety, env_name)
    env = gym.make(env_name)

    return env

def get_new_friction_env(variety, env_name):

    update_target_env_friction(variety, env_name)
    env = gym.make(env_name)

    return env


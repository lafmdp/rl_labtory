'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/25
'''

import os
import numpy as np
import gym
from multiprocessing import Pool
from multiprocessing import Process

from functions.sac_continuous import  policy, ReplayBuffer
from utils.utils import run_with_new_process, run_with_process_pool

np.set_printoptions(threshold=np.inf)

import tensorflow as tf
import  argparse
import time

if not os.path.exists("./Documents"):
    os.mkdir("./Documents")
if not os.path.exists("./Documents/PolicyModel"):
    os.mkdir("./Documents/PolicyModel")

parser = argparse.ArgumentParser(description="Running time configurations")

parser.add_argument('--env', default="Hopper-v2", type=str)
parser.add_argument('--vg', default="-1", type=str)
parser.add_argument('--process_num', default=10, type=int)
parser.add_argument('--points_num', default=5000, type=int)

args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = args.vg

process_num = args.process_num
points_per_process = args.points_num/process_num




def worker():

    pi = policy(have_model=True, need_log=False, action_space=act_dim, state_space=obs_dim)

    batch = {}
    for key in rl_keys:
        batch[key] = []

    point = 0
    while point < points_per_process:

        s = env.reset()

        traj_batch = {
            "state": [],
            "actions": [],
            "reward": [],
            "state_": [],
            "done":[]
        }

        step = 0

        while True:

            a = pi.get_action(s)

            s_, r, done, _ = env.step(a)


            traj_batch["state"].append(s)
            traj_batch["state_"].append(s_)
            traj_batch["reward"].append(r)
            traj_batch["actions"].append(a)
            traj_batch["done"].append(done)

            s = s_
            step += 1
            point += 1

            if done:

                batch["state"].append(traj_batch["state"])
                batch["state_"].append(traj_batch["state_"])
                batch["reward"].append(traj_batch["reward"])
                batch["action"].append(traj_batch["actions"])
                batch["done"].append(traj_batch["done"])
                batch["sum_reward"].append(sum(traj_batch["reward"]))
                batch["traj_len"].append(len(traj_batch["reward"]))
                batch["return"].append(pi.get_return(traj_batch["reward"]))

                break

    return batch

def train():

    pi = policy(have_model=True, need_log=True, action_space=act_dim, state_space=obs_dim)
    for _ in range(100):
        pi.train(buffer.sample_batch())
    pi.save_model()

def create_model():
    pi = policy(have_model=False, need_log=False, action_space=act_dim, state_space=obs_dim)
    pi.save_model()

if __name__ == "__main__":
    rl_keys = ["state", "action", "state_", "reward", "sum_reward", "done", "traj_len", "return"]

    env = gym.make(args.env)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])

    buffer = ReplayBuffer(obs_dim, act_dim, size=100000)

    print('Env Name:',args.env)
    print("obs_space:", obs_dim)
    print("action:", act_dim)

    run_with_new_process(create_model)

    iter = 0

    while True:

        t1 = time.time()
        iter += 1

        print('\n---------------------------- Iteration %d --------------------------------'%iter)

        batch = run_with_process_pool(worker, rl_keys, args.process_num)

        buffer.store(batch)
        print("sum reward:", np.hstack(batch["sum_reward"]).mean())
        print("average length:", np.hstack(batch["traj_len"]).mean())

        run_with_new_process(train)

        print("Time cosuming:",time.time()-t1)
        print('------------------------------------------------------------\n')

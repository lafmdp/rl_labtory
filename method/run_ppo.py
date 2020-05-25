'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/25
'''

import os
import numpy as np
import gym
from tensorboardX import  SummaryWriter
from multiprocessing import Pool, Process, Manager
from functions.ppo import policy
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
import  argparse
import time

if not os.path.exists("./Documents"):
    os.mkdir("./Documents")
if not os.path.exists("./Documents/PolicyModel"):
    os.mkdir("./Documents/PolicyModel")

parser = argparse.ArgumentParser(description="Running time configurations")

parser.add_argument('--env', default="Walker2d-v2", type=str)
parser.add_argument('--vg', default="-1", type=str)
parser.add_argument('--process_num', default=30, type=int)
parser.add_argument('--points_num', default=10000, type=int)
parser.add_argument('--seed', default=1, type=int)


args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = args.vg

rl_keys = ["state", "action", "state_", "reward", "gae", "return", "sum_reward", "trajectory_len"]

import random

# def set_global_seeds(i):
#     try:
#         from mpi4py import MPI
#         rank = MPI.COMM_WORLD.Get_rank()
#     except ImportError:
#         rank = 0
#
#     myseed = i  + 1000 * rank if i is not None else None
#     try:
#         import tensorflow as tf
#         tf.set_random_seed(myseed)
#     except ImportError:
#         pass
#     np.random.seed(myseed)
#     random.seed(myseed)
#
# set_global_seeds(10)

def worker(points_num, share_lock):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high

    pi = policy(have_model=True, need_log=False, action_space=act_dim, state_space=obs_dim)

    batch = {}
    for key in rl_keys:
        batch[key] = []

    point = 0
    while points_num.value <= args.points_num:

        s = env.reset()

        traj_batch = {
            "state": [],
            "actions": [],
            "reward": [],
            "gae": [],
            "value": []
        }

        step = 0

        while True:
            if points_num.value > args.points_num:
                break

            ret = pi.get_action(s)
            a = ret["actions"]

            s_, r, done, info = env.step(a * high[0])

            r *= (1-pi.gamma)

            traj_batch["state"].append(s)
            traj_batch["reward"].append(r)
            traj_batch["actions"].append(ret["actions"])
            traj_batch["value"].append(ret["value"])

            s = s_
            step += 1
            point += 1

            if done:

                v = pi.get_value(s_)
                real_next = traj_batch["value"][1:] + [np.array(v)]

                ret = pi.get_return(traj_batch["reward"])

                gae = pi.get_gaes(traj_batch["reward"], traj_batch["value"], real_next)

                batch["state"].append(traj_batch["state"])
                batch["reward"].append(traj_batch["reward"])
                batch["action"].append(traj_batch["actions"])
                batch["gae"].append(gae)
                batch["return"].append(ret)
                batch["trajectory_len"].append(len(traj_batch["state"]))
                # batch["sum_reward"].append(sum(traj_batch["reward"]))

                share_lock.acquire()
                points_num.value += len(traj_batch["state"])
                share_lock.release()

                break

    traj = 0
    while traj < 3:

        s = env.reset()

        traj_batch = {
            "reward": []
        }

        step = 0

        while True:

            a = pi.get_means(s)

            s_, r, done, info = env.step(a * high[0])

            traj_batch["reward"].append(r)

            s = s_
            step += 1

            if done:
                batch["sum_reward"].append(sum(traj_batch["reward"]))

                traj += 1
                break

    return batch



def train(batch):

    pi = policy(have_model=True, need_log=True, action_space=act_dim, state_space=obs_dim)
    pi.train(batch)
    pi.save_model()

def create_model():
    pi = policy(have_model=False, need_log=False, action_space=act_dim, state_space=obs_dim)
    pi.save_model()

if __name__ == "__main__":

    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print('Env Name:%s'%args.env)
    print("obs_space:", obs_dim)
    print("act_space:", act_dim)
    print("act_high:", env.action_space.high)
    print("act_low:", env.action_space.low)

    p = Process(target=create_model)
    p.start()
    p.join()

    iter = 0

    return_list = []
    import datetime
    writer = SummaryWriter("./tbfile/{}".format(datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')))

    for _ in range(2000):

        t1 = time.time()
        iter += 1

        print('\n---------------------------- Iteration %d --------------------------------'%iter)

        #batch = run_with_process_pool(worker, rl_keys, args.process_num)
        p = Pool(args.process_num)

        batch = {}

        for key in rl_keys:
            batch[key] = []

        results = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()

        for i in range(args.process_num):
            results.append(p.apply_async(worker, args=(points_num, share_lock,)))

        p.close()
        p.join()

        for res in results:

            res = res.get()
            for key in rl_keys:
                batch[key] += res[key]

        pro = Process(target=train, args=(batch,))
        pro.start()
        pro.join()
        first_step_return = np.array(batch["sum_reward"])
        return_list.append(first_step_return.mean())
        writer.add_scalar("sum_of_traj_reward", first_step_return.mean(), iter)

        print("Time cosuming:",time.time()-t1)
        print('------------------------------------------------------------\n')

    writer.close()

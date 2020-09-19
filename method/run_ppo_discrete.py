
import os
import numpy as np
import gym
from functions.ppo_discrete import policy
from tensorboardX import  SummaryWriter
from multiprocessing import Pool, Process, Manager
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
import argparse
import time

if not os.path.exists("./Documents"):
    os.mkdir("./Documents")
if not os.path.exists("./Documents/PolicyModel"):
    os.mkdir("./Documents/PolicyModel")

parser = argparse.ArgumentParser(description="Running time configurations")

parser.add_argument('--env', default="Kangaroo-ram-v0", type=str)
parser.add_argument('--vg', default="-1", type=str)
parser.add_argument('--process_num', default=10, type=int)
parser.add_argument('--points_num', default=10240, type=int)
parser.add_argument('--seed', default=1, type=int)


args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = args.vg

rl_keys = ["state", "action", "state_", "reward", "gae", "return", "sum_reward", "trajectory_len"]

def worker(points_num, share_lock):

    env.seed(os.getpid())
    pi = policy(have_model=True, action_space=act_dim, state_space=obs_dim)

    batch = {}
    for key in rl_keys:
        batch[key] = []

    point = 0
    while points_num.value <= args.points_num:

        s = env.reset()
        s = (s - mean) / std

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

            s_, r, done, info = env.step(a)
            s_ = (s_-mean)/std

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
                batch["sum_reward"].append(sum(traj_batch["reward"]))
                batch["trajectory_len"].append(len(traj_batch["state"]))

                share_lock.acquire()
                points_num.value += len(traj_batch["state"])
                share_lock.release()

                break

    # traj = 0
    # while traj < 3:
    #
    #     s = (env.reset()-mean)/std
    #
    #     traj_batch = {
    #         "reward": []
    #     }
    #
    #     step = 0
    #
    #     while True:
    #
    #         a = pi.get_means(s)
    #
    #         s_, r, done, info = env.step(a)
    #         s_ = (s_-mean)/std
    #
    #         traj_batch["reward"].append(r)
    #
    #         s = s_
    #         step += 1
    #
    #         if done:
    #             batch["sum_reward"].append(sum(traj_batch["reward"]))
    #
    #             traj += 1
    #             break

    return batch



def train(batch):

    pi = policy(have_model=True, action_space=act_dim, state_space=obs_dim)
    pi.train(batch)
    pi.save_model()

def create_model():
    pi = policy(have_model=False, action_space=act_dim, state_space=obs_dim)
    pi.save_model()


def try_env():
    state_list = []

    while len(state_list) < 1000:
        s = env.reset()

        while True:

            s,_,done,_ = env.step(env.action_space.sample())
            state_list.append(s)

            if done:
                break

    mean = np.mean(state_list,axis=0)
    std = np.std(state_list,axis=0)

    return mean, std


if __name__ == "__main__":

    env = gym.make(args.env)

    mean, std = try_env()
    std[std==0] = 1

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print('Env Name:%s'%args.env)
    print("obs_space:", obs_dim)
    print("act_space:", act_dim)
    print("Mean:",mean)
    print("Std:",std)

    p = Process(target=create_model)
    p.start()
    p.join()

    iter = 0

    return_list = []
    import datetime
    writer = SummaryWriter("./tbfile/{}".format(datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')))

    sample_cost = 0

    for _ in range(2000):

        t1 = time.time()
        iter += 1

        print('\n---------------------------- Iteration %d --------------------------------'%iter)

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
        sample_cost += np.vstack(batch["state"]).shape[0]
        writer.add_scalar("sum_of_traj_reward", first_step_return.mean(), sample_cost)

        print("Time cosuming:",time.time()-t1, "sample cost:", sample_cost)
        print('------------------------------------------------------------\n')

    writer.close()

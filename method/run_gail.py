'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/25
'''

import os
import numpy as np
from utils.utils import rl_keys
import gym
from multiprocessing import Pool
from multiprocessing import Process
from functions.gail import gail
from functions.existing_pi import existing_pi
from functions.replay_buffer import SAS_Buffer
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
import  argparse

parser = argparse.ArgumentParser(description="Running time configurations")

parser.add_argument('--env', default="Walker2d-v2", type=str)
parser.add_argument('--vg', default="1", type=str)
parser.add_argument('--process_num', default=20, type=int)
parser.add_argument('--points_num', default=10000, type=int)
parser.add_argument('--file_name', default="file.npy", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--reward', default="gail", type=str)
parser.add_argument('--horizon', default=3, type=int)



args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env_name = args.env
process_num = args.process_num
points_per_process = args.points_num/process_num
expert_episode_per_process = 10/process_num
np.random.seed(args.seed)

max = lambda a,b: a if a>b else b

def worker():

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high

    pi = gail(have_model=True, need_log=False, action_space=act_dim, state_space=obs_dim)

    batch = {}
    for key in rl_keys:
        batch[key] = []

    point = 0
    while point < points_per_process:

        s = env.reset()

        traj_batch = {
            "state": [],
            "actions": [],
            "state_":[],
            "reward": [],
            "gae": [],
            "value": [],
        }

        step = 0

        while True:

            ret = pi.get_action(s)
            a = ret["actions"]

            s_, env_r, done, _ = env.step(a * high[0])
            gail_r = pi.get_reward(s,s_)

            if args.reward=="gail":
                r = gail_r
            elif args.reward == "max":
                r = max(env_r, gail_r)
            elif args.reward == "plus":
                r = gail_r + env_r
            elif args.reward == "multiply":
                r = gail_r * env_r
            else:
                raise  RuntimeError("Wrong reward type")

            traj_batch["state"].append(s)
            traj_batch["reward"].append(r)
            traj_batch["state_"].append(s_)
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
                batch["state_"].append(traj_batch["state_"])
                batch["action"].append(traj_batch["actions"])
                batch["gae"].append(gae)
                batch["return"].append(ret)
                batch["trajectory_len"].append(len(traj_batch["state"]))

                break

    game_num = 0
    while game_num < 2:

        s = env.reset()

        sum_r = 0

        while True:

            a = pi.get_means(s)

            s_, r, done, _ = env.step(a * high[0])
            sum_r += r
            s = s_

            if done:

                game_num += 1
                batch["sum_reward"].append(sum_r)

                break

    return batch

def train(batch, iter):

    pi = gail(have_model=True, need_log=True, action_space=act_dim, state_space=obs_dim)
    pi.train(batch)

    gail_epoch = 1 if iter > 10 else 10

    for _ in range(gail_epoch):
        pi.train_discriminator(expert_samples, batch)

    pi.save_model()

def create_model():
    pi = gail(have_model=False, need_log=False, action_space=act_dim, state_space=obs_dim)
    pi.save_model()


def get_expert_samples():

    pi = existing_pi('./Documents/success/{}/policy'.format(args.env.split("-")[0].lower()))

    high = env.action_space.high

    batch = {
        "state":[],
        "action":[],
        "state_":[]
    }

    import  random
    seed = random.randint(0,10000)
    env.seed(seed)
    print('seed:',seed)


    game_num = 0
    while game_num < 1:

        traj_batch = {
            "state":[],
            "action":[],
            "state_":[]
        }

        s = env.reset()

        while True:

            a = pi.get_means(s)

            s_, r, done, _ = env.step(a * high[0])
            traj_batch["state"].append(s)
            traj_batch["action"].append(a)
            traj_batch["state_"].append(s_)
            s = s_

            if done:

                game_num += 1
                batch["state"].append(traj_batch["state"])
                batch["action"].append(traj_batch["action"])
                batch["state_"].append(traj_batch["state_"])

                # print(traj_batch["state"][:10])

                break

    return batch


if __name__ == "__main__":

    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    expert_samples = SAS_Buffer(obs_dim, act_dim, 20000, 512)

    print('Env Name:%s'%args.env)
    print("obs_space:", obs_dim)
    print("act_space:", act_dim)
    print("act_high:", env.action_space.high)
    print("act_low:", env.action_space.low)

    p = Process(target=create_model)
    p.start()
    p.join()

    print('-------------- Get expert samples --------------')
    p = Pool(args.process_num)

    expert_batch = {
        "state":[],
        "action":[],
        "state_":[]
    }

    results = []
    traj_num = 5 if args.env != "Humanoid-v2" else 50

    for i in range(traj_num):
        results.append(p.apply_async(get_expert_samples))

    p.close()
    p.join()

    for res in results:

        res = res.get()
        expert_batch["state"] += res["state"]
        expert_batch["action"] += res["action"]
        expert_batch["state_"] += res["state_"]

    print("expert sample num:",np.vstack(expert_batch["state"]).shape[0])

    expert_samples.append(expert_batch["state"], expert_batch["action"], expert_batch["state_"])

    iter = 0
    reward_list = []

    for _ in range(300):
        iter += 1

        print('\n-------------- Iteration %d --------------'%iter)

        p = Pool(process_num)

        batch = {}

        for key in rl_keys:
            batch[key] = []

        results = []

        for i in range(process_num):
            results.append(p.apply_async(worker))

        p.close()
        p.join()

        for res in results:

            res = res.get()
            for key in rl_keys:
                batch[key] += res[key]

        pro = Process(target=train, args=(batch, iter,))
        pro.start()
        pro.join()

        first_step_return = np.array(batch["sum_reward"])
        reward_list.append(first_step_return.mean())

    np.save(args.file_name, reward_list)

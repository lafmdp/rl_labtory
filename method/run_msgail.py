'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/25
'''

import os
import numpy as np
from utils.utils import rl_keys
import gym
from tensorboardX import SummaryWriter
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
parser.add_argument('--horizon', default=3, type=int)
parser.add_argument('--reward', default="gail", type=str)



args = parser.parse_args()

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env_name = args.env
process_num = args.process_num
points_per_process = args.points_num/process_num
expert_episode_per_process = 10/process_num
# np.random.seed(args.seed)
# tf.set_random_seed(args.seed)

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

            s_, _, done, _ = env.step(a * high[0])
            r = pi.get_reward(s,s_)

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


                new_reward_list = np.zeros_like(np.array(traj_batch["reward"]))
                traj_len = len(traj_batch["state"])

                state_list = []
                state__list = []
                complete_data_index = traj_len - args.horizon
                for index in range(traj_len):
                    if index <= complete_data_index:
                        for j in range(args.horizon):
                            state_ = traj_batch["state_"][index + j].copy().tolist()

                            state_list.append(traj_batch["state"][index])
                            state__list.append(state_)

                            batch["gail_state"].append(traj_batch["state"][index])
                            batch["gail_state_"].append(state_)
                    else:
                        state_ = traj_batch["state_"][index].copy().tolist()

                        state_list.append(traj_batch["state"][index])
                        state__list.append(state_)

                        batch["gail_state"].append(traj_batch["state"][index])
                        batch["gail_state_"].append(state_)

                D_reward = pi.get_batch_reward(state_list, state__list)

                for i in range(traj_len):
                    if i <= complete_data_index:
                        new_reward_list[i] = D_reward[(i * args.horizon):(
                                i * args.horizon + args.horizon)].max()
                    else:
                        new_reward_list[i] = D_reward[i - traj_len]

                traj_batch["reward"] = new_reward_list.tolist()

                ret = pi.get_return(traj_batch["reward"])

                gae = pi.get_gaes(traj_batch["reward"], traj_batch["value"], real_next)

                batch["state"].append(traj_batch["state"])
                batch["reward"].append(traj_batch["reward"])
                batch["state_"].append(traj_batch["state_"])
                batch["action"].append(traj_batch["actions"])
                batch["gae"].append(gae)
                batch["return"].append(ret)
                batch["D_reward"].append(sum(traj_batch["reward"]))
                batch["D_mean_reward"].append(sum(traj_batch["reward"])/len(traj_batch["reward"]))
                batch["trajectory_len"].append(len(traj_batch["state"]))

                break

    game_num = 0
    while game_num < 1:

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

    gail_epoch = 5 if iter > 10 else 20

    d_loss = []
    fa = []
    ra = []
    for _ in range(gail_epoch):
        ret = pi.train_discriminator(expert_samples, batch)
        d_loss.append(ret["d_loss"])
        fa.append(ret["fa"])
        ra.append(ret["ra"])

    pi.save_model()

    return dict(d_loss=np.array(d_loss).mean(),fa=np.array(fa).mean(), ra=np.array(ra).mean())

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
                traj_len = len(traj_batch["state"])
                for index in range(traj_len):
                    for i in range(3):
                        if index + i >= traj_len:
                            break

                        batch["state"].append(traj_batch["state"][index])
                        batch["action"].append(traj_batch["action"][index])
                        state_ = traj_batch["state_"][index + i].copy().tolist()
                        batch["state_"].append(state_)


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
    import datetime
    writer = SummaryWriter("./tbfile/{}".format(datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')))

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

        p = Pool(1)
        results = []

        for i in range(1):
            results.append(p.apply_async(train, args=(batch, iter,)))

        p.close()
        p.join()

        for res in results:
            ret = res.get()

        first_step_return = np.array(batch["sum_reward"])
        D_reward = np.array(batch["D_reward"])
        D_mean_reward = np.array(batch["D_mean_reward"])
        writer.add_scalar("reward/mujoco_reward", first_step_return.mean(),iter)
        writer.add_scalar("reward/D_reward", D_reward.mean(),iter)
        writer.add_scalar("reward/D_mean_reward", D_mean_reward.mean(),iter)
        writer.add_scalar("D/d_loss", ret["d_loss"],iter)
        writer.add_scalar("D/fake_accuracy", ret["fa"],iter)
        writer.add_scalar("D/real_accuracy", ret["ra"],iter)
        reward_list.append(first_step_return.mean())

    writer.close()

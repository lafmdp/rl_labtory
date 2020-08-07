'''
  Example for 2 players game PPO's use
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/11/21
'''


import numpy as np
from functions.preyandpredator import Game

from multiprocessing import Pool
from multiprocessing import Process
from functions.ppo_discrete import ppo
from collections import deque

np.set_printoptions(threshold=np.inf)
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

batch_keys = ["state", "action", "reward", "gae", "v_next", "mu_prob"]

process_num = 25
MAX_TRAJ = 100/process_num
obs_dim = 8
act_dim = 25

def worker():
    env = Game(5,5)

    prey = ppo(state_space=obs_dim, action_space=act_dim, have_model=True, p_type="prey")
    dator = ppo(state_space=obs_dim, action_space=act_dim, have_model=True, p_type="dator")


    prey_batch = {}
    dator_batch = {}

    for key in batch_keys:
        prey_batch[key] = []
        dator_batch[key] = []

    traj_num = 0

    while True:

        s = env.reset()

        prey_traj = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
            "mu_prob":[]
        }

        dator_traj = {
            "state": [],
            "action": [],
            "reward": [],
            "value": [],
            "mu_prob": []
        }

        step = 0
        traj_num += 1

        while True:
            s = np.array(s)

            a_prey, prob_prey, v_prey = prey.get_action_value(s)
            a_dator, prob_dator, v_dator = dator.get_action_value(s)

            #print("nn output",a_dator)

            f1 = int(a_dator/5)
            f2 = a_dator % 5
            #print('f1,f2',f1,f2)

            g1 = int(a_prey/5)
            g2 = a_prey % 5

            s_, r_prey, r_dator = env.step(f1,f2,g1,g2)

            r_prey *= 0.01
            r_dator *= 0.01

            prey_traj["state"].append(s)
            prey_traj["reward"].append(r_prey)
            prey_traj["action"].append(a_prey)
            prey_traj["value"].append(v_prey)
            prey_traj["mu_prob"].append(prob_prey)

            dator_traj["state"].append(s)
            dator_traj["reward"].append(r_dator)
            dator_traj["action"].append(a_dator)
            dator_traj["value"].append(v_dator)
            dator_traj["mu_prob"].append(prob_dator)

            s = s_
            step += 1

            if step > 100:
                v_prey = prey.get_value(s_)
                v_prey_next = prey_traj["value"][1:] + [np.array(v_prey)]

                gae = prey.PPOTrain.get_gaes(prey_traj["reward"], prey_traj["value"], v_prey_next)

                prey_batch["state"].append(prey_traj["state"])
                prey_batch["action"].append(prey_traj["action"])
                prey_batch["reward"].append(prey_traj["reward"])
                prey_batch["gae"].append(gae)
                prey_batch["v_next"].append(v_prey_next)
                prey_batch["mu_prob"].append(prey_traj["mu_prob"])

                v_dator = dator.get_value(s_)
                v_dator_next = dator_traj["value"][1:] + [np.array(v_dator)]

                gae = dator.PPOTrain.get_gaes(dator_traj["reward"], dator_traj["value"], v_dator_next)

                dator_batch["state"].append(dator_traj["state"])
                dator_batch["action"].append(dator_traj["action"])
                dator_batch["reward"].append(dator_traj["reward"])
                dator_batch["gae"].append(gae)
                dator_batch["v_next"].append(v_dator_next)
                dator_batch["mu_prob"].append(dator_traj["mu_prob"])

                break

        if traj_num >= MAX_TRAJ:
            return {
                "prey_batch":prey_batch,
                "dator_batch":dator_batch
            }


def train(batch_prey, batch_dator):
    prey = ppo(state_space=obs_dim, action_space=act_dim, have_model=True, p_type="prey")
    dator = ppo(state_space=obs_dim, action_space=act_dim, have_model=True, p_type="dator")

    prey.train(batch_prey)
    dator.train(batch_dator)
    prey.save_model()
    dator.save_model()

def create_model():
    prey = ppo(state_space=obs_dim, action_space=act_dim, have_model=False, p_type="prey")
    dator = ppo(state_space=obs_dim, action_space=act_dim, have_model=False, p_type="dator")

    prey.save_model()
    dator.save_model()

if __name__ == "__main__":

    env = Game(5,5)

    p = Process(target=create_model)
    p.start()
    p.join()

    prey_reward = deque(maxlen=5)
    dator_reward = deque(maxlen=5)
    iteration = 0

    prey_list = []
    dator_list = []

    while True:

        iteration += 1

        print('----------iteration %d----------'%iteration)

        p = Pool(process_num)

        prey_batch = {}
        dator_batch = {}

        for key in batch_keys:
            prey_batch[key] = []
            dator_batch[key] = []

        results = []

        for i in range(process_num):
            results.append(p.apply_async(worker))

        p.close()
        p.join()

        for res in results:

            res = res.get()
            bforp = res["prey_batch"]
            bford = res["dator_batch"]

            for key in batch_keys:
                prey_batch[key] += bforp[key]
                dator_batch[key] += bford[key]


        reward_prey = np.vstack(prey_batch["reward"])
        prey_reward.append(reward_prey.sum(axis=1).mean())

        reward_dator = np.vstack(dator_batch["reward"])
        dator_reward.append(reward_dator.sum(axis=1).mean())

        pro = Process(target=train, args=(prey_batch, dator_batch,))
        pro.start()
        pro.join()

        prey_list.append(sum(prey_reward)/len(prey_reward))
        dator_list.append(sum(dator_reward)/len(dator_reward))
        print('prey return',sum(prey_reward)/len(prey_reward))
        print('dator return', sum(dator_reward)/len(dator_reward))

        if iteration % 10 is 0:
            prey_data = np.array(prey_list)
            dator_data = np.array(dator_list)

            np.save("dator.npy", dator_data)
            np.save("prey.npy", prey_data)

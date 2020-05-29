'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/2/7
'''

'''

  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/29
'''

import os
import gym#; print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))
import numpy as np
import random
from functions.customized_mujuco import self_mujuco
from utils.replay_buffer import SAS_Buffer
from utils.utils import *
from functions.existing_pi import existing_pi
from functions.behavior_cloning import behavior_cloning
from functions.inverse_dynamic import inverse_dynamic
from multiprocessing import Pool, Process, Manager

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

process_num = 10


key_list = [
    "state","action","state_",
    "ms_state","ms_action","ms_state_"
]

import  argparse

parser = argparse.ArgumentParser(description="Running time configurations")

parser.add_argument('--env', default="Walker2d-v2", type=str)
parser.add_argument('--vg', default="1", type=str)
parser.add_argument('--process_num', default=20, type=int)
parser.add_argument('--points_num', default=10000, type=int)
parser.add_argument('--file_name', default="file.npy", type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--expert_traj_num', default=1, type=int)
parser.add_argument('--horizon', default=5, type=int)

args = parser.parse_args()
model_path = './Documents/success/{}/policy'.format(args.env.lower().split("-")[0])


result = []

def sampleEnv(points_num, share_lock):

    pi = existing_pi(model_path=model_path)

    batch = {
        "state": [],
        "action": [],
        "state_": [],
        "sum_reward": []
    }

    while True:

        s = env.reset()

        traj_batch = {
            "state": [],
            "action": [],
            "state_": [],
            "reward": []
        }

        while True:
            if points_num.value > args.expert_traj_num:
                return batch

            means = pi.get_means(s)

            s_, r, done, _ = env.step(means * high)

            r *= 0.01
            traj_batch["state"].append(s)
            traj_batch["action"].append(means)
            traj_batch["state_"].append(s_)
            traj_batch["reward"].append(r)

            s = s_

            if done:
                batch["state"] += traj_batch["state"]
                batch["action"] += traj_batch["action"]
                batch["state_"] += traj_batch["state_"]
                batch["sum_reward"].append(sum(traj_batch["reward"]))
                share_lock.acquire()
                points_num.value += 1
                share_lock.release()

                break

def sampleMSEnv(points_num, share_lock):

    pi = existing_pi(model_path=model_path)

    batch = {}
    for key in key_list:
        batch[key] = []

    while True:

        s = env.reset()

        traj_batch = {
            "state": [],
            "action": [],
            "state_": []
        }
        sum_reward = 0

        while True:
            if points_num.value >= args.expert_traj_num:
                return batch

            means = pi.get_means(s)

            s_, r, done, _ = env.step(means * high)
            sum_reward += r

            traj_batch["state"].append(s)
            traj_batch["action"].append(means)
            traj_batch["state_"].append(s_)
            s = s_

            if done:
                if sum_reward < 2000:
                    break
                else:
                    share_lock.acquire()
                    points_num.value += 1
                    share_lock.release()

                    traj_len = len(traj_batch["state"])
                    for index in range(traj_len):
                        for i in range(args.horizon):
                            if index + i >= traj_len:
                                break

                            batch["ms_state"].append(traj_batch["state"][index])
                            batch["ms_action"].append(traj_batch["action"][index])
                            state_ = traj_batch["state_"][index + i].copy().tolist()
                            batch["ms_state_"].append(state_)

                    batch["state"].append(traj_batch["state"])
                    batch["state_"].append(traj_batch["state_"])
                    batch["action"].append(traj_batch["action"])

                    print("Expert Traj Reward:",sum_reward)

                    break



def trainIdWorker():
    """
    Train inverse dynamic model for expert on the target.
    """

    idt = inverse_dynamic(model_path=idm_path,
                          have_model=False,
                          state_dim=obs_dim,
                          act_dim=act_dim)

    id_data = id_expert_samples.collect_all_samples()

    for i in range(20000):
        idt.train(id_expert_samples.specific_sample(id_data))

    print("Trainning id in 20000 epoch(s)")

    idt.save_model()


def trainBcWorker():
    """
    Train inverse dynamic model for expert on the target.
    """

    bcm = behavior_cloning(model_path=idm_path,
                          have_model=False,
                          state_dim=obs_dim,
                          act_dim=act_dim)

    id_data = bc_expert_samples.collect_all_samples()

    for _ in range(20000):
        bcm.train(bc_expert_samples.specific_sample(id_data))

    print("Trainning bc in 20000 epoch(s)")

    bcm.save_model()


def initNetworks():

    idm_target = inverse_dynamic(state_dim=obs_dim,
                                 act_dim=act_dim,
                                 model_path=idm_path,
                                 have_model=False)

    idm_target.save_model()

    bcm = behavior_cloning(state_dim=obs_dim,
                                 act_dim=act_dim,
                                 model_path=idm_path,
                                 have_model=False)

    bcm.save_model()


def evaluateWorker(perform_type: str, eva_num, share_lock):
    """
    evaluate
    :param perform_type: target or source
    :param policy_type: expert or policy
    :return:
    """

    if perform_type == "id":
        po = existing_pi(model_path=model_path)
        idm = inverse_dynamic(obs_dim,act_dim, idm_path,have_model=True)

    elif perform_type == "bc":
        po = behavior_cloning(state_dim=obs_dim,
                               act_dim=act_dim,
                               model_path=idm_path,
                               have_model=True)
    else:
        po = existing_pi(model_path=model_path)

    batch = {
        "sum_reward": []
    }

    game_num = 0

    while True:
        game_num += 1

        s = env.reset()

        sum_reward = 0

        while True:
            if eva_num.value > 100:
                return batch

            if perform_type == "id":
                virtual_action = po.get_action(s)
                s_pred,_,_,_ = ground_truth_transition.forward(env.sim.get_state(), virtual_action)#+np.random.normal(0, std,virtual_action.shape))
                # s_pred = np.array(s_pred)
                # s_pred = np.random.normal(s_pred, std, s_pred.shape)
                ret = idm.get_action(s, s_pred)
                a = ret["actions"]
            else:
                a = po.get_means(s)

            s_, r, done, _ = env.step(a * high[0])

            sum_reward += r
            s = s_

            if done:
                batch["sum_reward"].append(sum_reward)

                share_lock.acquire()
                eva_num.value += 1
                share_lock.release()

                break

def evaluate(perform_type):

    p = Pool(process_num)

    batch = {
        "sum_reward": []
    }

    eva_num = Manager().Value("l", 0)
    share_lock = Manager().Lock()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    results = []

    for i in range(process_num):
        results.append(p.apply_async(evaluateWorker, args=(perform_type, eva_num, share_lock,)))

    p.close()
    p.join()

    for res in results:
        res = res.get()
        batch["sum_reward"] += res["sum_reward"]

    rt = np.hstack(batch["sum_reward"]).astype(dtype=np.float32).squeeze()

    print("-------------- Performance summary --------------")
    print("Type:", perform_type)
    print("High:{},\tlow:{},\nMean:{},\tstd:{}".format(rt.max(),
                                                       rt.min(),
                                                       rt.mean(),
                                                       rt.std()))

    result.append(rt.mean())


if __name__ == "__main__":

    env = gym.make("Walker2d-v2")
    ground_truth_transition = self_mujuco("Walker2d-v2")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high
    act_space = env.action_space

    idm_path = "./tmp"

    for traj_num in range(1, 11, 2):
        print("------------trajnum:%d-------------"%3)
        args.expert_traj_num = 3

        run_with_new_process(initNetworks)

        id_expert_samples = SAS_Buffer(obs_dim, act_dim, size=1, batch_size=300)
        bc_expert_samples = SAS_Buffer(obs_dim, act_dim, size=1, batch_size=300)

        p = Pool(process_num)

        batch = {}
        for key in key_list:
            batch[key] = []

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()
        results = []

        for i in range(process_num):
            results.append(p.apply_async(sampleMSEnv, args=(points_num, share_lock,)))

        p.close()
        p.join()

        for res in results:
            res = res.get()
            for key in key_list:
                batch[key] += res[key]

        id_expert_samples.append(batch["ms_state"], batch["ms_action"], batch["ms_state_"])
        bc_expert_samples.append(batch["state"], batch["action"], batch["state_"])

        run_with_new_process(trainBcWorker)
        run_with_new_process(trainIdWorker)
        print("Observe inverse dynamic performance")
        evaluate("id")

        print("Observe behavior cloning performance")
        evaluate("bc")
        print("--------------------------------------------------\n\n")

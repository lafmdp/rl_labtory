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

process_num = 25
model_path = './Documents/success/walker2d/policy'

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
            if points_num.value > 5:
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

def trainIdWorker():
    """
    Train inverse dynamic model for expert on the target.
    """

    idt = inverse_dynamic(model_path=idm_path,
                          have_model=False,
                          state_dim=obs_dim,
                          act_dim=act_dim)

    id_data = expert_target_samples.collect_all_samples()
    sample_num = id_data.shape[0]
    train_index = np.random.choice(sample_num, round(sample_num * 0.8), replace=False).tolist()
    test_index = np.array(list(set(range(sample_num)) - set(train_index)))

    train_data = id_data[train_index]
    test_data = id_data[test_index]

    i = 0
    while i < 15000:
        idt.train(expert_target_samples.specific_sample(id_data))

        i += 1

    print("Trainning id in %d epoch(s)" % (i))

    idt.save_model()


def trainBcWorker():
    """
    Train inverse dynamic model for expert on the target.
    """

    bcm = behavior_cloning(model_path=idm_path,
                          have_model=False,
                          state_dim=obs_dim,
                          act_dim=act_dim)

    id_data = expert_target_samples.collect_all_samples()
    sample_num = id_data.shape[0]
    train_index = np.random.choice(sample_num, round(sample_num * 0.8), replace=False).tolist()

    train_data = id_data[train_index]

    i = 0
    while i < 15000:
        bcm.train(expert_target_samples.specific_sample(train_data))

        i += 1

    print("Trainning bc in %d epoch(s)" % (i))

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
            if eva_num.value > 50:
                return batch

            if perform_type == "id":
                virtual_action = po.get_means(s)
                s_pred,_,_,_ = ground_truth_transition.forward(env.sim.get_state(), virtual_action)#+np.random.normal(0, std,virtual_action.shape))
                s_pred = np.array(s_pred)
                s_pred = np.random.normal(s_pred, std, s_pred.shape)
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

    # print("Observe expert performance")
    # evaluate("policy")

    for traj_num in range(10):
        print("\n\n----------------- tranjectory nums:%d -----------------\n" % (traj_num+1))

        run_with_new_process(initNetworks)
        std = traj_num/5

        expert_target_samples = SAS_Buffer(obs_dim, act_dim, size=1, batch_size=300)

        p = Pool(process_num)

        batch = \
            {
                "state": [],
                "action": [],
                "state_": []
            }

        points_num = Manager().Value("l", 0)
        share_lock = Manager().Lock()
        results = []

        for i in range(process_num):
            results.append(p.apply_async(sampleEnv, args=(points_num, share_lock,)))

        p.close()
        p.join()

        for res in results:
            res = res.get()
            batch["state"] += res["state"]
            batch["action"] += res["action"]
            batch["state_"] += res["state_"]

        expert_target_samples.append(batch["state"], batch["action"], batch["state_"])

        # expert_target_samples.buffer = np.load("./data.npy")[:sample_num,:]

        run_with_new_process(trainIdWorker)
        # run_with_new_process(trainBcWorker)

        print("Observe inverse dynamic performance")
        evaluate("id")

        # print("Observe behavior cloning performance")
        # evaluate("bc")
        print("--------------------------------------------------")

    np.save("./result_traj_1.npy", result)


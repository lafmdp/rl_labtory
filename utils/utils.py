'''
  
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/12/9
'''

import copy
import numpy as np
from tensorflow.python.framework import graph_util
import  tensorflow as tf

rl_keys = ["state", "action", "state_", "reward", "gae", "return", "sum_reward", "trajectory_len", "gail_state","gail_state_", "D_reward", "D_mean_reward"]

hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "need_log": False,
    "batch_size": 5000,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 3,
    "c_2": 0.001,
    "init_lr": 3e-4,
    "d_lr": 1e-3,
    "lr_epsilon": 1e-6
}

def set_policy(graph_batch):
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph_batch["pb_string"])
    graph = tf.Graph()


    # load now policy model from bytes
    with graph.as_default():
        tf.import_graph_def(graph_def, name='Policy')

        # load placeholders
        inputs = graph.get_tensor_by_name('Policy/' + graph_batch["state_ph"] + ':0')

        # load operations
        actions = graph.get_tensor_by_name('Policy/' + graph_batch["action_op"] + ':0')
        values = graph.get_tensor_by_name('Policy/' + graph_batch["value_op"] + ':0')

    sess = tf.Session(graph=graph)

    def build_policy(state):
        with sess.as_default(), graph.as_default():
            obs = np.array(state)

            if obs.shape[0] != 1:
                obs = obs[np.newaxis, :]

            action, value = \
                sess.run([actions, values], feed_dict={inputs: obs})

            ret = {
                "action": action[0],
                "value": value[0]
            }

        return ret

    return build_policy

def get_return(rewards):
    dis_rewards = np.zeros_like(rewards).astype(np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * hype_parameters["gamma"] + rewards[t]
        dis_rewards[t] = running_add

    return dis_rewards


def get_gaes(rewards, v_preds, v_preds_next):

    deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

    return gaes

def run_with_new_process(func, *args):
    from multiprocessing import Process
    pro = Process(target=func, args=(args))
    pro.start()
    pro.join()


def run_with_process_pool(func, keys, process_num, *args):
    from multiprocessing import Pool

    p = Pool(process_num)

    batch = {}

    for key in keys:
        batch[key] = []

    results = []

    for i in range(process_num):
        results.append(p.apply_async(func, args=(args)))

    p.close()
    p.join()

    for res in results:

        res = res.get()
        for key in keys:
            batch[key] += res[key]

    return batch

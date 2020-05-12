'''
  Utils functions and some configs.
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/9/23
'''

import re
import os
import copy
import time
import random
import argparse
import datetime
import numpy as np
import tensorflow as tf



nowTime = datetime.datetime.now().strftime('%y-%m-%d%H:%M:%S')

parser = argparse.ArgumentParser(description="Process running arguments")

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

hype_parameters = {
    "gamma": 0.99,
    "lamda": 0.95,
    "need_log": False,
    "batch_size": 4096,
    "epoch_num": 10,
    "clip_value": 0.2,
    "c_1": 1,
    "c_2": 0.001,
    "init_lr": 3e-4,
    "d_lr": 5e-4,
    "lr_epsilon": 1e-6
}


# add argument if needed
parser.add_argument('--process_num', default=32, type=int, help='process numbers')
parser.add_argument('--vg', default="1", type=str, help='visible gpus')
parser.add_argument('--discription', default="None", type=str, help='extra discription added to log file')
parser.add_argument('--file_name', default="result.npy", type=str)
parser.add_argument('--env_list',default="Walker2d-v2", nargs='+', help='avaliable env')

parser.add_argument('--points_num', default=1, type=int, help='sample nums for all')
parser.add_argument('--eva_num', default=20, type=int, help='evaluation game nums each process')
parser.add_argument('--expert_iter', default=1, type=int, help='iteration nums for training an expert')
parser.add_argument('--nowtime', default=nowTime, type=str, help='current time for file name')
parser.add_argument('--variety_low', default=1, type=float)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def generate_xml_path():
    import sys
    path = sys.path

    xml_path = None

    for p in path:

        if "site-packages" in p[-14:] and "local" not in p:
            xml_path = p + '/gym/envs/mujoco/assets'
            print(xml_path)

    assert xml_path is not None

    return xml_path

gym_xml_path = generate_xml_path()

def record_data(file, content):
    with open(file, 'a+') as f:
        f.write('{}\n'.format(content))


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

def update_xml(index, env_name):

    xml_name = parse_xml_name(env_name)
    os.system('cp ./xml_path/{0}/{1} {2}/{1}}'.format(index, xml_name, gym_xml_path))

    time.sleep(0.2)


def parse_xml_name(env_name):
    if 'walker' in env_name.lower():
        xml_name = "walker2d.xml"
    elif 'double' in env_name.lower():
        xml_name = "inverted_double_pendulum.xml"
    elif 'hopper' in env_name.lower():
        xml_name = "hopper.xml"
    elif 'reach' in env_name.lower():
        xml_name = "reacher.xml"
    elif 'halfcheetah' in env_name.lower():
        xml_name = "half_cheetah.xml"
    elif "standup" in env_name.lower():
        xml_name = "humanoidstandup.xml"
    elif "ant" in env_name.lower():
        xml_name = "ant.xml"
    elif "striker" in env_name.lower():
        xml_name = "striker.xml"
    elif "swim" in env_name.lower():
        xml_name = "swimmer.xml"
    elif "throw" in env_name.lower():
        xml_name = "thrower.xml"
    elif "point" in env_name.lower():
        xml_name = "point.xml"
    elif "pendulum" in env_name.lower():
        xml_name = "inverted_pendulum.xml"
    else:
        raise RuntimeError("No available env named \'%s\'"%env_name)

    return xml_name


def update_source_env(env_name):
    xml_name = parse_xml_name(env_name)

    os.system(
        'cp ./xml_path/source_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)


def update_target_env_gear(variety_degree, env_name):

    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gear" in line:
                pattern = re.compile(r'(?<=gear=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)

def update_target_env_gravity(variety_degree, env_name):

    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "gravity" in line:
                pattern = re.compile(r"gravity=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree*float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "gravity=\""+replace_num+"\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)

def update_target_env_density(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "density" in line:
                pattern = re.compile(r'(?<=density=")\d+\.?\d*')
                a = pattern.findall(line)
                current_num = float(a[0])
                replace_num = current_num * variety_degree
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)

def update_target_env_friction(variety_degree, env_name):
    xml_name = parse_xml_name(env_name)

    with open('./xml_path/source_file/{}'.format(xml_name), "r+") as f:

        new_f = open('./xml_path/target_file/{}'.format(xml_name), "w")
        for line in f.readlines():
            if "friction" in line:
                pattern = re.compile(r"friction=\"(.*?)\"")
                a = pattern.findall(line)
                friction_list = a[0].split(" ")
                new_friction_list = []
                for num in friction_list:
                    new_friction_list.append(variety_degree*float(num))

                replace_num = " ".join(str(i) for i in new_friction_list)
                replace_num = "friction=\""+replace_num+"\""
                sub_result = re.sub(pattern, str(replace_num), line)

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system(
        'cp ./xml_path/target_file/{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)


def generate_log(extra=None):
    print(extra)
    record_data('log.txt', "{}".format(extra))


def get_gaes(rewards, v_preds, v_preds_next):

    deltas = [r_t + hype_parameters["gamma"] * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + hype_parameters["gamma"] * hype_parameters["lamda"] * gaes[t + 1]

    return gaes

def get_return(rewards):

    dis_rewards = np.zeros_like(rewards).astype(np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * hype_parameters["gamma"] + rewards[t]
        dis_rewards[t] = running_add

    return dis_rewards


def set_global_seeds(i):
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.set_random_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)


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
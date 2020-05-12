'''

  @python version : 3.6.8
  @author : pangjc
  @time : 2019/8/29
'''

import os
import gym#; print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))
import numpy as np
from functions.existing_pi import existing_pi
from functions.existing_gail_pi import existing_gail_pi
from functions.customized_mujuco import self_mujuco

import argparse
parser = argparse.ArgumentParser(description="Running time configurations")
parser.add_argument('--env', default="Walker2d-v2", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def generate_xml_path():
    import sys
    path = sys.path

    xml_path = None

    for p in path:
        if "site-packages" in p[-14:] and "local" not in p:
            xml_path = p + '/gym/envs/mujoco/assets'

    assert xml_path is not None

    return xml_path


def update_target_env_gravity(variety_degree):
    import os
    import time

    xml_name = "walker2d.xml"
    gym_xml_path = generate_xml_path()
    print(gym_xml_path)

    with open('./walker2d.xml'.format(xml_name), "r+") as f:

        new_f = open('./test.xml'.format(xml_name), "w")
        for line in f.readlines():
            if "-9.81" in line:
                replace_num = -9.81 * variety_degree
                sub_result = line.replace("-9.81", str(replace_num))

                new_f.write(sub_result)
            else:
                new_f.write(line)

        new_f.close()

    os.system("mv ./test.xml walker2d.xml")

    os.system(
        'cp ./{0} {1}/{0}'.format(xml_name, gym_xml_path))

    time.sleep(0.5)


if __name__ == "__main__":
    env = gym.make(args.env)
    simulator = self_mujuco(args.env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high
    act_space = env.action_space

    # result_dir = './Documents/PolicyModel/'
    # lists = os.listdir(result_dir)
    # lists.sort(key=lambda fn: os.path.getmtime(result_dir + '/' + fn))

    # po = existing_pi(model_path='./Documents/PolicyModel/%s/policy' % lists[-1])
    # po = existing_gail_pi(model_path='./Documents/PolicyModel/{}/gail'.format(lists[-1]))
    dir = args.env.lower().split("-")[0]


    po = existing_pi(model_path='./Documents/success/{}/policy'.format(dir))


    r_list = []
    len_list = []
    game_num = 0

    real_state_list = []
    real_r_list = []
    sim_state_list  = []
    sim_r_list = []

    while True:

        s = env.reset()
        sum_r = 0
        len_this = 0

        while True:

            # env.render()
            a = po.get_means(s)
            simulator.set_sim_state(env.sim.get_state())
            s_sim, r_sim, _, _ = simulator.step(a)
            s_, r, done, info = env.step(a * high[0])

            real_state_list.append(s_)
            real_r_list.append(r)
            sim_state_list.append(s_sim)
            sim_r_list.append(r_sim)

            sum_r += r
            len_this += 1

            s = s_

            if done:

                game_num += 1
                r_list.append(sum_r)
                len_list.append(len_this)

                print("sum_r:{}, average:{}, average_len:{}".format(sum_r, sum(r_list)/len(r_list), sum(len_list)/len(len_list)))

                # print("\tsim\treal\t")
                # for i in range(len(sim_r_list)):
                #     print("Step:",i)
                #     print("\t{}\t{}\t".format(sim_state_list[i], real_state_list[i]))
                #     print("\t{}\t{}\t".format(sim_r_list[i], real_r_list[i]))
                #     print()

                break


    # update_target_env_gravity(-2)
    # env = gym.make("Walker2d-v2")

    # game_num = 0
    #
    # while game_num < 1:
    #
    #     s = env.reset()
    #     sum_r = 0
    #
    #     while True:
    #         env.render()
    #         a = po.get_action(s)
    #
    #         s_, r, done, _ = env.step(a * high[0])
    #         sum_r += r
    #
    #         s_list.append(s)
    #
    #         s = s_
    #
    #         if done:
    #             game_num += 1
    #             state_list = np.array(s_list)
    #             print(sum_r)
    #
    #             break

'''

  @python version : 3.6.4
  @author : pangjc
  @time : 2020/5/25
'''

from functions.ppo import policy
from utils.envs import *
from utils.utils_new import *
import numpy as np
import gym
from multiprocessing import Pool, Process, Manager

# add argument if needed
parser.add_argument('--process_num', default=30, type=int, help='process numbers')
parser.add_argument('--env_list',default="Walker2d-v2;Ant-v2;Hopper-v2;HalfCheetah-v2", type=str, help='avaliable env, seperated by ;')
parser.add_argument('--transfer_type',default="friction;gravity", type=str)
parser.add_argument('--points_num', default=50000, type=int)
parser.add_argument('--iteration_num', default=150, type=int)
parser.add_argument('--variety_list', default="0.5;1.5;2.0", type=str)



args = parser.parse_args()


class ppo_wrapper():


    def __init__(self):
        self.rl_keys = ["state", "action", "state_", "reward", "gae", "return", "sum_reward", "trajectory_len"]

    def set_training_configuration(self, env, model_path):
        self.env = env
        self.model_path = model_path
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.high = env.action_space.high

    def see_performance_worker(self):
        pi = policy(model_path=self.model_path, have_model=True, need_log=False, action_space=self.act_dim, state_space=self.obs_dim)

        batch = {}
        for key in self.rl_keys:
            batch[key] = []

        traj = 0
        while traj < 3:

            s = self.env.reset()

            traj_batch = {
                "reward": []
            }

            step = 0

            while True:

                a = pi.get_means(s)

                s_, r, done, info = self.env.step(a * self.high[0])

                traj_batch["reward"].append(r)

                s = s_
                step += 1

                if done:
                    batch["sum_reward"].append(sum(traj_batch["reward"]))

                    traj += 1
                    break

        return batch


    def worker(self, points_num, share_lock):

        pi = policy(model_path=self.model_path, have_model=True, need_log=False, action_space=self.act_dim, state_space=self.obs_dim)

        batch = {}
        for key in self.rl_keys:
            batch[key] = []

        point = 0
        while True:

            s = self.env.reset()

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
                    return batch

                ret = pi.get_action(s)
                a = ret["actions"]

                s_, r, done, info = self.env.step(a * self.high[0])

                r *= (1 - pi.gamma)

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
                    batch["sum_reward"].append(sum(traj_batch["reward"]))

                    share_lock.acquire()
                    points_num.value += len(traj_batch["state"])
                    share_lock.release()

                    break

    def train(self, batch):

        pi = policy(model_path=self.model_path, have_model=True, need_log=True, action_space=self.act_dim, state_space=self.obs_dim)
        pi.train(batch)
        pi.save_model()

    def create_model(self):
        pi = policy(model_path=self.model_path, have_model=False, need_log=False, action_space=self.act_dim, state_space=self.obs_dim)
        pi.save_model()

    def run(self):

        p = Process(target=self.create_model)
        p.start()
        p.join()

        iter = 0


        for _ in range(args.iteration_num):

            iter += 1

            p = Pool(args.process_num)

            batch = {}

            for key in self.rl_keys:
                batch[key] = []

            results = []

            points_num = Manager().Value("l", 0)
            share_lock = Manager().Lock()

            for i in range(args.process_num):
                results.append(p.apply_async(self.worker, args=(points_num, share_lock,)))

            p.close()
            p.join()

            for res in results:

                res = res.get()
                for key in self.rl_keys:
                    batch[key] += res[key]

            pro = Process(target=self.train, args=(batch,))
            pro.start()
            pro.join()

        self.see_performance()


    def see_performance(self):
        p = Pool(args.process_num)

        batch = {"sum_reward":[]}

        results = []

        for i in range(args.process_num):
            results.append(p.apply_async(self.see_performance_worker))

        p.close()
        p.join()

        for res in results:

            res = res.get()
            batch["sum_reward"] += res["sum_reward"]

        print("Performance:", np.array(batch["sum_reward"]).mean())

if __name__ == '__main__':
    check_path("./expert")

    ppo = ppo_wrapper()

    for transfer_type in args.transfer_type.split(";"):
        check_path("./expert/{}".format(transfer_type))
        for env_name in args.env_list.split(";"):
            check_path("./expert/{}/{}".format(transfer_type,env_name))

            for variety_degree in args.variety_list.split(";"):
                variety_degree = float(variety_degree)

                print("\n----------{}----------{}----------{}----------".format(transfer_type,env_name,variety_degree))

                check_path("./expert/{}/{}/{}".format(transfer_type, env_name, variety_degree))

                if transfer_type == "gravity":
                    env = get_new_gravity_env(variety_degree, env_name)
                elif transfer_type == "density":
                    env = get_new_density_env(variety_degree, env_name)
                elif transfer_type == "friction":
                    env = get_new_friction_env(variety_degree, env_name)
                else:
                    raise RuntimeError("Got error transfer type %s" % transfer_type)

                model_path = "./expert/{}/{}/{}".format(transfer_type, env_name, variety_degree)
                ppo.set_training_configuration(env, model_path)
                ppo.run()

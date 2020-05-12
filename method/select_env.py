'''
  
  @python version : 3.6.4
  @author : pangjc
  @time : 2020/2/7
'''


from utils.utils_new import *
from utils.envs import *
from functions.existing_pi import existing_pi
from functions.ppo import policy

from multiprocessing import Pool, Process, Manager

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

rl_keys = ["state", "action", "state_", "reward", "gae", "return", "sum_reward", "trajectory_len"]


def worker():

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    high = env.action_space.high

    pi = policy(have_model=True, need_log=False, action_space=act_dim, state_space=obs_dim)

    batch = {}
    for key in rl_keys:
        batch[key] = []

    point = 0
    points_per_process = args.points_num/args.process_num
    while point < points_per_process:

        s = env.reset()

        traj_batch = {
            "state": [],
            "actions": [],
            "reward": [],
            "gae": [],
            "value": [],

        }

        step = 0

        while True:

            ret = pi.get_action(s)
            a = ret["actions"]

            s_, r, done, info = env.step(a * high[0])

            r *= (1-pi.gamma)

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

                break

    return batch

def train(batch):

    pi = policy(have_model=True,
                need_log=True,
                action_space=act_dim,
                state_space=obs_dim)
    pi.train(batch)
    pi.save_model()

def create_model():
    pi = policy(have_model=False, need_log=False, action_space=act_dim, state_space=obs_dim)
    pi.save_model()


def evaluateWorker(perform_type: str, eva_num, share_lock):
    """
    evaluate
    :param perform_type: target or source
    :param policy_type: expert or policy
    :return:
    """

    if perform_type == "expert":
        pi = policy(have_model=True, need_log=False, action_space=act_dim, state_space=obs_dim)
    else:
        pi = existing_pi(model_path='./Documents/success/%s/policy' % env_name.lower())

    batch = {
        "sum_reward": []
    }

    game_num = 0

    while True:
        game_num += 1

        s = env.reset()

        sum_reward = 0

        while True:
            if eva_num.value > 200:
                return batch

            a = pi.get_means(s)

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

    p = Pool(args.process_num)

    batch = {
        "sum_reward": []
    }

    eva_num = Manager().Value("l", 0)
    share_lock = Manager().Lock()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    results = []

    for i in range(args.process_num):
        results.append(p.apply_async(evaluateWorker, args=(perform_type, eva_num, share_lock,)))

    p.close()
    p.join()

    for res in results:
        res = res.get()
        batch["sum_reward"] += res["sum_reward"]

    rt = np.hstack(batch["sum_reward"]).astype(dtype=np.float32).squeeze()

    generate_log("Type:{},Mean:{},\tstd:{}".format(perform_type,
                                           rt.mean(),
                                           rt.std()))

if __name__ == "__main__":


    for env_name in ["Walker2d", "Hopper", "HalfCheetah", "Reacher"]:
        for transfer_type in ["density", "gravity", "friction"]:
            for vd in [0.5, 0.8, 1, 1.2, 1.6, 2.0, 2.2, 2.5, 3.0]:

                generate_log("\n-------------------- {}-{}-{} ----------------------".format(env_name, transfer_type, vd))

                full_env_name = "{}-v2".format(env_name)
                if transfer_type == "density":
                    env = get_new_density_env(vd, full_env_name)
                elif transfer_type == "friction":
                    env = get_new_friction_env(vd, full_env_name)
                else:
                    env = get_new_gravity_env(vd, full_env_name)

                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.shape[0]
                high = env.action_space.high
                act_space = env.action_space

                p = Process(target=create_model)
                p.start()
                p.join()

                # # train a expert on env
                # for i in range(args.expert_iter):
                #     batch = run_with_process_pool(worker, rl_keys, args.process_num)
                #
                #     pro = Process(target=train, args=(batch,))
                #     pro.start()
                #     pro.join()

                # evaluate source and expert
                # run_with_new_process(evaluate, "expert")
                run_with_new_process(evaluate, "policy")


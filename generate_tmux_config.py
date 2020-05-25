import argparse
import time

import yaml
# python generate_tmux_yaml.py --num_seeds 4 --env_names "Hopper-v2"
parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument(
    '--num_seeds',
    type=int,
    default=3,
    help='number of random seeds to generate')

parser.add_argument(
    '--process_num',
    type=int,
    default=5,
    help='process Nums per panel')


parser.add_argument(
    '--conda_name',
    default="pangjc",
    type=str,
    help='conda activate if needed')

args = parser.parse_args()

run_template = "python -m method.run_ppo " \
               "--process_num {} " \
               "--seed {} "


template = run_template
dir_name = 0
sleep_inverval = 1

config = {"session_name": "compare-gail", "windows": []}

for reward_type in ["max"]:

    panes_list = []
    for i in range(args.num_seeds):


        pane_str = template.format(args.process_num, i*1000)
        dir_name += 1
        pane_str = "sleep {}s && ".format(sleep_inverval*i) + pane_str

        if args.conda_name is not None:
            pane_str = "source activate {} && ".format(args.conda_name)+pane_str

        panes_list.append(pane_str)

    config["windows"].append({
        "window_name": "{}".format(reward_type),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)

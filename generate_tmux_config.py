import argparse
import time

import yaml
# python generate_tmux_yaml.py --num_seeds 4 --env_names "Hopper-v2"
parser = argparse.ArgumentParser(description='Process some parameters.')


parser.add_argument(
    '--process_num',
    type=int,
    default=10,
    help='process Nums per panel')

parser.add_argument(
    '--conda_name',
    default="pangjc",
    type=str,
    help='conda activate if needed')

parser.add_argument(
    '--transfer_type',
    default="friction",
    type=str)

parser.add_argument(
    '--env_list',
    default="Walker2d-v2;Ant-v2;Hopper-v2;HalfCheetah-v2",
    type=str)


args = parser.parse_args()

run_template = "python -m method.generate_expert_policy " \
               "--process_num {} " \
               "--iteration_num 200 " \
               "--env_list {}" \
               "--transfer_type {}"

template = run_template
dir_name = 0
sleep_inverval = 1

config = {"session_name": "generate_expert", "windows": []}

for env_name in args.env_list.split(";"):

    panes_list = []

    pane_str = template.format(args.process_num,
                               env_name,
                               args.transfer_type)
    dir_name += 1
    pane_str = "sleep {}s && ".format(sleep_inverval*i) + pane_str

    if args.conda_name is not None:
        pane_str = "source activate {} && ".format(args.conda_name)+pane_str

    panes_list.append(pane_str)

    config["windows"].append({
        "window_name": "{}".format(env_name),
        "panes": panes_list
    })

yaml.dump(config, open("run_all.yaml", "w"), default_flow_style=False)

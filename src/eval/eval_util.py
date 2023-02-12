import copy
import json
import datetime
from pathlib import Path
import config
import os
from os import listdir
from os.path import isfile, join
import sys


def store_eval_output(normalized_states: list, states: list, model_output: list, state_format: dict, player_idx: int,
                      episode_number: int):
    """
    store the eval output for entire round
    :param normalized_states:
    :param states:
    :param model_output:
    :param state_format:
    :param player_idx:
    :param episode_number:
    :return:
    """
    path = "{}/eval/{}".format(config.settings['data_path'], config.settings['run_name'])
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    states = copy.deepcopy(states)
    for idx, v in enumerate(states):
        states[idx]['input'] = states[idx]['input'][player_idx]

    all_data = {
        "normalized_states": normalized_states,
        "states": copy.deepcopy(states),
        "model_output": copy.deepcopy(model_output),
        "state_format": state_format
    }
    Path("{}/evals/{}/{}".format(path, player_idx, episode_number)).mkdir(parents=True, exist_ok=True)
    Path("{}/stats/{}".format(path, player_idx)).mkdir(parents=True, exist_ok=True)

    with open("{}/evals/{}/{}/eval_{}.json".format(path, player_idx, episode_number, datetime_str), 'a') as f_writer:
        f_writer.write(json.dumps(all_data))

    with open("{}/config.json".format(path), 'w') as f_writer:
        f_writer.write(json.dumps(config.settings))


def get_reward_paths(player_idx):
    reward_path = "{}/eval/{}/reward/{}".format(config.settings['data_path'], config.settings['run_name'], player_idx)
    reward_paths = []

    if os.path.exists(reward_path):
        reward_paths = ["{}/{}".format(reward_path, f) for f in listdir(reward_path) if
                        not isfile(join(reward_path, f))]

    return reward_paths


def get_next_episode(player_idx):
    eval_path = "{}/eval/{}/evals/{}".format(config.settings['data_path'], config.settings['run_name'], player_idx)

    max_episode = 0
    run_count = 1
    if os.path.exists(eval_path):
        episode_dirs = [f for f in listdir(eval_path) if not isfile(join(eval_path, f))]
        print("episode_dirs={}".format(episode_dirs))

        for d in episode_dirs:
            if d.isnumeric():
                if int(d) > max_episode:
                    max_episode = int(d)
        print("max_episode={}".format(max_episode))

        episode_dirs = [f for f in listdir(eval_path) if not isfile(join(eval_path, f))]

        for d in episode_dirs:
            if d.isnumeric():
                if int(d) > max_episode:
                    max_episode = int(d)

        episode_path = "{}/{}".format(eval_path, max_episode)

        if os.path.exists(episode_path):
            episode_files = [f for f in listdir(episode_path) if isfile(join(episode_path, f))]
            print(len(episode_files))

            if config.settings['count_save'] == len(episode_files):
                max_episode = max_episode + 1
            else:
                run_count = len(episode_files) + 1
        else:
            print("no path {}".format(episode_path))

    return max_episode, run_count


def print_q(action, q):
    total_len = 30
    bar_str = "["
    bar_len = 0

    if q is not None:
        min_q = -7
        max_q = 3

        norm_q = (max_q - q) / (max_q - min_q)
        r_q = round(norm_q, 2)
        num = round(total_len - (r_q * total_len))

        for _ in range(0, num):
            bar_str = bar_str + "="
            bar_len = bar_len + 1
        q = round(q,2)
        norm_q = round(norm_q, 2)
    else:
        q = "*"
        norm_q = "*"

    for _ in range(bar_len, total_len):
        bar_str = bar_str + " "

    bar_str = bar_str + "]"

    sys.stdout.write("\r action= {} est_norm_Q= {} est_Q= {} {}".format(
        str(action).ljust(3),
        str(norm_q).ljust(4),
        str(q).ljust(3),
        bar_str))
    sys.stdout.flush()

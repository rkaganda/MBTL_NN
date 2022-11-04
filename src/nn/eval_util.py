import copy
import json
import datetime
from functools import singledispatch
from pathlib import Path
import config
import torch
import os
from os import listdir
from os.path import isfile, join


def store_eval_output(normalized_states, states, model_output, state_format, player_idx, episode_number):
    path = "data/eval/{}".format(config.settings['run_name'])
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
    Path("{}/model/{}/{}".format(path, player_idx, episode_number)).mkdir(parents=True, exist_ok=True)
    Path("{}/stats/{}".format(path, player_idx)).mkdir(parents=True, exist_ok=True)

    with open("{}/evals/{}/{}/eval_{}.json".format(path, player_idx, episode_number, datetime_str),'a') as f_writer:
        f_writer.write(json.dumps(all_data))

    with open("{}/config.json".format(path),'w') as f_writer:
        f_writer.write(json.dumps(config.settings))


def get_next_episode(player_idx):
    eval_path = "data/eval/{}/evals/{}".format(config.settings['run_name'], player_idx)

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







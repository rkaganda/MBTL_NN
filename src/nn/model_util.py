import torch

from os import listdir
from os.path import isfile, join
import json
from pathlib import Path
import os
import datetime

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


def load_model_config(p_idx: int) -> dict:
    model_config = {}

    dir_path = "{}/models/{}".format(
        config.settings['data_path'], config.settings["p{}_model".format(p_idx)]['name'])
    path = "{}/model_config.json".format(dir_path)
    if os.path.exists(path) and isfile(path):
        with open(path) as f:
            model_config = json.load(f)
    else:
        print("no p{} model_config.json, using config.yaml".format(p_idx))
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        model_config_yaml = config.settings["p{}_model".format(p_idx)]
        for k_, v_ in model_config_yaml.items():
            model_config[k_] = v_
        with open(path, "w") as f_writer:
            f_writer.write(json.dumps(model_config))

    return model_config


def get_episode_from_path(path):
    max_episode = 0
    if os.path.exists(path):
        episode_dirs = [f for f in listdir(path) if not isfile(join(path, f))]

        for d in episode_dirs:
            if d.isnumeric():
                if int(d) > max_episode:
                    max_episode = int(d)

    return max_episode


def load_model(model, optimizer, player_idx):
    path = "{}/models/{}".format(
        config.settings['data_path'], config.settings["p{}_model".format(player_idx)]['name'])

    ep_num = get_episode_from_path(path)
    path = "{}/{}".format(path, ep_num)

    if os.path.exists(path):
        model_files = [f for f in listdir(path) if isfile(join(path, f))]
        model_path = None
        optim_path = None
        for mf in model_files:
            if mf.endswith(".model"):
                model_path = "{}/{}".format(path, mf)
            else:
                optim_path = "{}/{}".format(path, mf)

        print("loading model={}".format(path))
        if model_path is not None and optim_path is not None:
            model.load_state_dict(torch.load(model_path, map_location=device))
            optimizer.load_state_dict(torch.load(optim_path, map_location=device))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print("loaded model = {}".format(model_path))
            print("loaded optimizer = {}".format(optim_path))

            return True

    return False


def save_model(model, optim, player_idx):
    path = "{}/models/{}".format(
        config.settings['data_path'], config.settings["p{}_model".format(player_idx)]['name'])

    ep_num = get_episode_from_path(path)
    path = "{}/{}".format(path, ep_num+1)

    Path(path).mkdir(parents=True, exist_ok=True)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save(model.state_dict(), "{}/{}.model".format(path, time_str))
    print("{}/{}.model saved".format(path, time_str))
    torch.save(optim.state_dict(), "{}/{}.optim".format(path, time_str))
    print("{}/{}.optim saved".format(path, time_str))
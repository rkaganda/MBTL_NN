import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as td
from torch.autograd import Variable
import config
import datetime
from pathlib import Path
import numpy as np
from os import listdir
from os.path import isfile, join
import json


# class Model(eval.Module):
#     def __init__(self, input_size, output_size):
#         super(Model, self).__init__()
#         # self.flatten = eval.Flatten()
#         self.linear_relu_stack = eval.Sequential(
#             eval.Linear(input_size, 512),
#             eval.ReLU(),
#             eval.Linear(512, 512),
#             eval.ReLU(),
#             eval.Linear(512, 512),
#             eval.ReLU(),
#             eval.Linear(512, 512),
#             eval.ReLU(),
#             eval.Linear(512, output_size),
#         )
#         self.activation = torch.eval.Sigmoid()
#
#     def forward(self, x):
#         # x = self.flatten(x)
#         out = self.linear_relu_stack(x)
#         return self.activation(out)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # self.flatten = eval.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
        # self.activation = torch.eval.ReLU()

    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
        # return self.activation(out)


def setup_model(frames_per_observation, input_lookback, input_state_size, state_state_size, learning_rate):
    input_layer_size = (frames_per_observation * (state_state_size * 2)) + input_lookback
    print("input_layer_size={}".format(input_layer_size))
    model = Model(input_layer_size, input_state_size)
    model.share_memory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer


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
        model_config['frames_per_observation'] = int(model_config_yaml['frames_per_observation'])
        model_config['input_lookback'] = int(model_config_yaml['input_lookback'])
        model_config['reaction_delay'] = int(model_config_yaml['reaction_delay'])
        model_config['learning_rate'] = float(model_config_yaml['learning_rate'])
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


def load_model(model, optimizer, player_idx, device):
    # path = "{}/eval/{}/model/{}/{}".format(config.settings['data_path'], config.settings['run_name'], player_idx, ep_num)
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
    # path = "{}/eval/{}/model/{}/{}/".format(config.settings['data_path'], config.settings['run_name'], player_idx, episode_num)

    path = "{}/models/{}".format(
        config.settings['data_path'], config.settings["p{}_model".format(player_idx)]['name'])

    ep_num = get_episode_from_path(path)
    path = "{}/{}".format(path, ep_num+1)

    Path(path).mkdir(parents=True, exist_ok=True)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save(model.state_dict(), "{}/{}.model".format(path, time_str))
    torch.save(optim.state_dict(), "{}/{}.optim".format(path, time_str))


# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

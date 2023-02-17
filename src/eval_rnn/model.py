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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, num_layers):
        super(Model, self).__init__()

        # init params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # layers
        # rn
        self.rnn = nn.RNN(input_size, hidden_dim, num_layers, batch_first=True)
        # fc layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # init for first input
        hidden = self.init_hidden(batch_size)

        # pass input and hidden state to run
        out, hidden = self.rnn(x, hidden)

        # reshape for fc
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # create zero for first pass hidden
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)

        return hidden


def setup_model(input_size, actions_size, learning_rate):
    model = Model(
        input_size=input_size,
        output_size=actions_size,
        hidden_dim=input_size,
        num_layers=2
    )
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
    torch.save(optim.state_dict(), "{}/{}.optim".format(path, time_str))




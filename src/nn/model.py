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


# class Model(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Model, self).__init__()
#         # self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_size),
#         )
#         self.activation = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         # x = self.flatten(x)
#         out = self.linear_relu_stack(x)
#         return self.activation(out)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(.20),
            nn.Linear(512, output_size),
        )
        # self.activation = torch.nn.ReLU()

    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return out
        # return self.activation(out)


def setup_model(frames_per_observation, input_state_size, state_state_size, learning_rate):
    input_layer_size = frames_per_observation * (1 + (state_state_size * 2))
    model = Model(input_layer_size, input_state_size)
    model.share_memory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer


def load_model(model, optimizer, player_idx, episode_number, device):
    for ep_num in reversed(range(-1, episode_number - 1)):
        path = "data/eval/{}/model/{}/{}".format(config.settings['run_name'], player_idx, ep_num)

        model_files = [f for f in listdir(path) if isfile(join(path, f))]
        model_path = None
        optim_path = None
        for mf in model_files:
            if mf.endswith(".model"):
                model_path = "{}/{}".format(path, mf)
            else:
                optim_path = "{}/{}".format(path, mf)

        if model_path is None:
            print("no model in episode={} going to prev".format(ep_num))
            continue

        print("loading model={}".format(path))

        model.load_state_dict(torch.load(model_path, map_location=device))
        optimizer.load_state_dict(torch.load(optim_path))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded model = {}".format(model_path))
        print("loaded optimizer = {}".format(optim_path))
        break

    return model, optimizer


def save_model(model, optim, player_idx, episode_num):
    path = "data/eval/{}/model/{}/{}/".format(config.settings['run_name'], player_idx, episode_num)

    Path(path).mkdir(parents=True, exist_ok=True)

    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print("saving model {}".format(time_str))

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

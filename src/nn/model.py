import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as td
from torch.autograd import Variable
import config
import datetime


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        # x = self.flatten(x)
        out = self.linear_relu_stack(x)
        return self.activation(out)


def setup_model(frames_per_observation, input_state_size, state_state_size, learning_rate):
    input_layer_size = frames_per_observation*(input_state_size+(state_state_size*2))
    model = Model(input_layer_size, input_state_size)
    model.share_memory()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer


def load_model(model, optimizer, player_idx):
    path = "data/eval/{}/model/{}".format(config.settings['run_name'], player_idx)

    model.load_state_dict(
        torch.load("{}/{}.model".format(path, config.settings['model_file']),
                   map_location=lambda storage, loc: storage))
    optimizer.load_state_dict(torch.load("{}/{}.optim".format(path, config.settings['model_file'])))
    print("loaded model = {}".format("{}/{}.model".format(path, config.settings['model_file'])))


def save_model(model, optim, player_idx):
    path = "data/eval/{}/model/".format(config.settings['run_name'])
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    torch.save(model.state_dict(), "{}/{}.model".format(path, time_str))
    torch.save(optim.state_dict(), "{}/{}.optim".format(path, time_str))





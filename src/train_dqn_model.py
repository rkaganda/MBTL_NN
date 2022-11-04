import copy
import logging

import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import json
import numpy as np

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

import time
import datetime

import config

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
#             nn.Linear(512, output_size),
#         )
#         self.activation = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         # x = self.flatten(x)
#         out = self.linear_relu_stack(x)
#         return self.activation(out)
#
#
# def setup_model(input_layer_size, output_layer_size, learning_rate):
#     model = Model(input_layer_size, output_layer_size)
#     model.share_memory()
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
#     return model, optimizer
#
#
# def load_model(model, optimizer):
#     path = "data/eval"
#     model.load_state_dict(
#         torch.load("{}/{}/model/{}.model".format(path, run_name, load_model_file),
#                    map_location=lambda storage, loc: storage))
#     optimizer.load_state_dict(torch.load("{}/{}/model/{}.optim".format(path, run_name, load_model_file)))

def load_reward_data(reward_paths):
    all_files = []
    for r_path in reward_paths:
        onlyfiles = [f for f in listdir(r_path) if isfile(join(r_path, f))]
        for f in onlyfiles:
            if f.startswith('reward_'):
                all_files.append("{}/{}".format(r_path, f))

    full_data = {
        'action': [],
        'state': [],
        'reward': [],
        'next_state': [],
        'done': []
    }
    for file in tqdm(all_files):
        with open(file) as f:
            file_dict = json.load(f)
        for k_, i_ in file_dict.items():
            full_data[k_] = full_data[k_] + i_
        next_state = copy.deepcopy(file_dict['state'])
        next_state = next_state[1:]
        next_state.append([0]*len(next_state[0]))
        full_data['next_state'] = full_data['next_state'] + next_state
        done = [0] * len(next_state)
        done[-1] = 1
        full_data['done'] = full_data['done'] + done
        break
    return full_data


def create_dataset(data):
    state_tensor = torch.Tensor(data['state']).to(device)
    action_tensor = torch.Tensor(data['action']).to(device)
    reward_tensor = torch.Tensor(data['reward']).to(device)
    next_state_tensor = torch.Tensor(data['next_state']).to(device)
    done = torch.Tensor(data['done']).to(device)

    return td.TensorDataset(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)


def train(model, target, optim, criterion, data, batch_size):
    state = Variable(data[0])
    next_states = Variable(data[3])
    actions = Variable(data[1])
    rewards = Variable(data[2])
    done = Variable(data[4])

    model.train()
    train_loss = 0

    # pred_q = model(state)
    #
    # pred_q = pred_q.gather(1, actions.type(torch.int64))

    pred_q = model(state).gather(1, actions.type(torch.int64))
    next_state_q_vals = torch.zeros(batch_size).to(device)

    for idx, next_state in enumerate(next_states):
        if done[idx] == 1:
            next_state_q_vals[idx] = -1
        else:
            # .max in pytorch returns (values, idx), we only want vals
            next_state_q_vals[idx] = (target(next_states[idx]).max(0)[0])

    better_pred = (rewards + next_state_q_vals).unsqueeze(1)

    loss = F.smooth_l1_loss(pred_q, better_pred).to(device)
    optim.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
        param.grad.to(device)
    optim.step()

    train_loss += loss.item()

    return train_loss, optim.get_last_lr()


def train_model(reward_paths, stats_path, model, target, optim, epochs, episode_num):
    reward_data = load_reward_data(reward_paths)
    batch_size = config.settings['batch_size']

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)

    dataset = create_dataset(reward_data)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    stats = {}
    step_count = 0

    for epoch in tqdm(range(0, epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, lr = train(model, target, optim, criterion, batch_data, batch_size)
            stats[step] = {
                "batch_size": len(batch_data[0]),
                "loss": train_loss,
                "learning rate": lr
            }
            break

    with open("{}/{}.json".format(stats_path, episode_num), 'a') as f_writer:
        f_writer.write(json.dumps(stats))


# def main():
#     run_name = 'p1_x_posi_test_1'
#     eval_path = "data/eval/{}".format(run_name)
#     learning_rate = 1e-5
#     epochs = 100
#     load_model_file = '20221022_003718'
#
#     reward_data = load_reward_data()
#
#     input_size = len(reward_data['input'][0])
#     output_size = len(reward_data['output'][0])
#
#     model, optim = setup_model(input_size, output_size, learning_rate)
#     criterion = nn.CrossEntropyLoss(reduction='none')
#     model.to(device)
#
#     print(model)
#     print(optim)
#
#     dataset = create_dataset(reward_data)
#
#     # N = len(dataset)
#     # data_sampler = td.sampler.RandomSampler(range(N))
#     # dataloader = td.DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
#
#     # with torch.profiler.profile(
#     #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
#     #         on_trace_ready=torch.profiler.tensorboard_trace_handler('../../logs/{}_{}'.format(dt_str, run_name)),
#     #         record_shapes=True,
#     #         profile_memory=True,
#     #         with_stack=True,
#     #         activities=[
#     #             torch.profiler.ProfilerActivity.CPU,
#     #             torch.profiler.ProfilerActivity.CUDA,
#     #         ]
#     # ) as prof:
#     for epoch in tqdm(range(0, epochs)):
#         for step, batch_data in enumerate(train_loader):
#             train(model, optim, criterion, batch_data)
#                 # train(batch_data)
#                 #prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.
#
#     print("{} done".format(dt_str))
#
#     save_model(model, optim)
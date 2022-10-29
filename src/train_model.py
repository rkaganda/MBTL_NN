import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable

import json
import numpy as np

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

import time
import datetime

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


def load_reward_data(reward_path):
    onlyfiles = [f for f in listdir(reward_path) if isfile(join(reward_path, f))]

    full_data = {
        'input': [],
        'output': [],
        'reward': []
    }

    for file in tqdm(onlyfiles):
        if file.startswith('reward_'):
            with open("{}/{}".format(reward_path, file)) as f:
                file_dict = json.load(f)
            for k_, i_ in file_dict.items():
                full_data[k_] = full_data[k_] + i_
    return full_data


# def save_model(model, optim, model_path):
#     dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     torch.save(model.state_dict(), "{}/model/{}.model".format(model_path, dt_str))
#     torch.save(optim.state_dict(), "{}/optim/{}.optim".format(model_path, dt_str))
#
#     print("{}.model saved".format(dt_str))
#     print("{}.optim saved".format(dt_str))


def create_dataset(data):
    input_tensor = torch.Tensor(data['input']).to(device)
    output_tensor = torch.Tensor(data['output']).to(device)
    reward_tensor = torch.Tensor(data['reward']).to(device)

    return td.TensorDataset(input_tensor, output_tensor, reward_tensor)


def train(model, optim, criterion, data):
    model.train()

    observation = Variable(data[0]).to(device)
    action_out = model(observation).to(device)

    # actions = Variable(t[:, 0].type(torch.LongTensor))
    # rewards = Variable(t[:, 1].type(torch.LongTensor))
    actions = Variable(data[1]).to(device)
    rewards = Variable(data[2]).to(device)

    # Calculates the loss for the movement outputs and the attack outputs
    loss = torch.sum(rewards * (criterion(actions, action_out)))

    optim.zero_grad()  # Resets the models gradients
    loss.backward()
    optim.step()  # Updates the network weights based on the calculated gradients


def train_model(reward_path, model, optim, epochs):
    reward_data = load_reward_data(reward_path)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)

    print(model)
    print(optim)

    dataset = create_dataset(reward_data)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in tqdm(range(0, epochs)):
        for step, batch_data in enumerate(train_loader):
            train(model, optim, criterion, batch_data)


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
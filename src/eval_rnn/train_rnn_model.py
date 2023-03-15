import copy
import logging

import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import json

from os import listdir
from os.path import isfile, join

from tqdm import tqdm

import config

logging.basicConfig(filename='../../logs/train.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        full_data['next_state'] = full_data['state'][1:]
        full_data['next_state'].append([0.0] * len(full_data['next_state'][0]))

    return full_data


class RollingDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards, next_states, done, window):
        self.data = []

        index = 0
        while index < len(done) - window:
            self.data.append([
                torch.Tensor(states[index: index + window]),
                torch.Tensor(actions[index: index + window]),
                torch.Tensor(rewards[index: index + window]),
                torch.Tensor(next_states[index: index + window]),
                torch.Tensor(done[index: index + window])
            ])

            if done[index + window - 1] == 0:
                index = index + 1
            else:
                index = index + window

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def create_dataset(data, window_size):
    state_tensor = data['state']
    action_tensor = data['action']
    reward_tensor = data['reward']
    next_state_tensor = data['next_state']
    done_tensor = data['done']
    dataset = RollingDataset(
        states=state_tensor,
        actions=action_tensor,
        rewards=reward_tensor,
        next_states=next_state_tensor,
        done=done_tensor,
        window=window_size)

    return dataset


def train(model, target, optim, data):
    state = Variable(data[0]).to(device)
    action = Variable(data[1]).to(device)
    reward = Variable(data[2]).to(device)
    next_state = Variable(data[3]).to(device)
    done = Variable(data[4]).to(device)

    model.train()
    gamma = .5

    with torch.no_grad():
        next_q_values, _ = target(next_state)
        next_q_values = next_q_values.max(dim=1)[0]

        reward = torch.flatten(reward)

        done = torch.flatten(done)

        q_targets = reward + (1 - done) * next_q_values * config.settings['reward_gamma']

    q_values, _ = model(state)

    action = action.flatten().unsqueeze(1)

    q_values = q_values.gather(1, action.to(torch.int64)).squeeze(1)
    reward_error = (q_values - q_targets).abs().mean()

    loss = F.smooth_l1_loss(q_values, q_targets)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optim.step()

    return loss, reward_error, optim.param_groups[0]['lr']


def train_model(reward_paths, stats_path, model, target, optim, epochs, episode_num, window_size):
    writer = SummaryWriter(stats_path)
    reward_data = load_reward_data(reward_paths)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)

    dataset = create_dataset(reward_data, window_size)
    sampler = torch.utils.data.RandomSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

    eps_loss = 0.0
    total_reward_error = 0.0
    for epoch in tqdm(range(epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, reward_error, lr = train(model, target, optim, batch_data)
            eps_loss = eps_loss + train_loss
            total_reward_error = reward_error + total_reward_error
            # if step >= config.settings['batch_size']:
            #     break
    writer.add_scalar("Reward Error/train", total_reward_error/epochs, episode_num)
    writer.add_scalar("Loss/train", eps_loss/epochs, episode_num)
    writer.flush()



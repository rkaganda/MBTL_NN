import logging
import torch
import torch.nn as nn
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

import json
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

logging.basicConfig(filename='../../logs/train.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_reward_file_paths(reward_paths):
    all_files = []
    for r_path in reward_paths:
        onlyfiles = [f for f in listdir(r_path) if isfile(join(r_path, f))]
        for f in onlyfiles:
            if f.startswith('reward_'):
                all_files.append("{}/{}".format(r_path, f))

    return all_files


class FullRollingDataset(torch.utils.data.Dataset):
    def __init__(self, reward_paths, window):
        self.window = window
        self.data = {
            'action': None,
            'state': None,
            'reward': None,
            'next_state': None,
            'done': None
        }
        for file in tqdm(reward_paths, position=0, leave=True):
            with open(file) as f:
                file_dict = json.load(f)
            for k_, i_ in file_dict.items():
                if k_ in self.data.keys():
                    if self.data[k_] is None:
                        self.data[k_] = i_
                    else:
                        self.data[k_] = self.data[k_] + i_
                next_state = file_dict['state'][1:].copy()
                next_state.append([0.0] * len(file_dict['state'][0]))
                if self.data['next_state'] is None:
                    self.data['next_state'] = next_state
                else:
                    self.data['next_state'] = self.data['next_state'] + next_state
        for k_ in self.data.keys():
            self.data[k_] = torch.Tensor(self.data[k_]).to(device)

    def __getitem__(self, index):
        return [
            self.data['action'][index:index + self.window],
            self.data['action'][index:index + self.window],
            self.data['reward'][index:index + self.window],
            self.data['next_state'][index:index + self.window],
            self.data['done'][index:index + self.window]
        ]

    def __len__(self):
        return len(self.data['state']) - self.window - 1


def train(model, target, optim, model_type, criterion, data, gamma):
    # state = data[0].to(device)
    # action = data[1].to(device)
    # reward = data[2].to(device)
    # next_state = data[3].to(device)
    # done = data[4].to(device)
    state = data[0]
    action = data[1]
    reward = data[2]
    next_state = data[3]
    done = data[4]

    if model_type == 'transformer':
        state = state.transpose(0, 1)  # reshape to (seq_length, batch_size, features)
        next_state = next_state.transpose(0, 1)  # reshape (seq_length, batch_size, features)

    model.train()

    with torch.no_grad():
        if model_type == 'rnn':
            # rnn returns last out and hidden state
            next_q_values, _ = target(next_state)
            next_q_values, _ = next_q_values.max(dim=1)
            reward = reward.flatten()
            done = done.flatten()
        elif model_type == 'transformer':
            # transformer returns full sequence
            next_q_values = target(next_state)
            next_q_values, _ = next_q_values.max(dim=2)
            reward = reward.transpose(0, 1)
            done = done.transpose(0, 1)

        q_targets = reward + (1 - done) * next_q_values * gamma
        if model_type == 'transformer':
            q_targets = q_targets.transpose(0, 1)  # transpose for loss

    if model_type == 'rnn':
        # rnn returns last out and hidden state
        q_values, _ = model(state)
        action = action.flatten()
        q_values = q_values.gather(1, action.to(torch.int64).unsqueeze(1)).squeeze(1)
    elif model_type == 'transformer':
        # transformer returns full sequence
        q_values = model(state)
        action = action.transpose(0, 1).squeeze(1)
        q_values = q_values.gather(2, action.to(torch.int64)).squeeze(2)
        q_values = q_values.transpose(0, 1)  # transpose for loss

    mean_reward_error = (q_values - q_targets).abs().mean()
    max_reward_error = torch.max((q_values - q_targets))
    std_reward_error = (q_values - q_targets).std()

    loss = criterion(q_values, q_targets)
    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optim.step()

    return loss, mean_reward_error, max_reward_error, std_reward_error, optim.param_groups[0]['lr']


def train_model(reward_paths, stats_path, model, target, optim, model_type, epochs, episode_num, window_size, gamma):
    print(reward_paths)
    writer = SummaryWriter(stats_path)

    criterion = nn.SmoothL1Loss()
    criterion.to(device)
    model.to(device)

    dataset = FullRollingDataset(get_reward_file_paths(reward_paths), window_size)
    sampler = torch.utils.data.RandomSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

    eps_loss = 0
    total_steps = 0
    total_reward_error = 0
    total_std_reward_error = 0
    max_error = 0
    for epoch in tqdm(range(epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, reward_error, batch_max_error, std_reward_error,  lr = train(model, target, optim, model_type, criterion, batch_data, gamma)
            eps_loss = eps_loss + train_loss
            total_reward_error = total_reward_error + reward_error
            total_std_reward_error = std_reward_error + std_reward_error
            total_steps = total_steps + 1
            max_error = max([max_error, batch_max_error])
    writer.add_scalar("LR/train", optim.param_groups[0]['lr'], episode_num)
    writer.add_scalar("Loss/train", eps_loss / (epochs), episode_num)
    writer.add_scalar("Reward Error/mean", total_reward_error / (epochs), episode_num)
    writer.add_scalar("Reward Error/std", total_std_reward_error / (epochs), episode_num)
    writer.add_scalar("Reward Error/max", max_error, episode_num)
    writer.flush()
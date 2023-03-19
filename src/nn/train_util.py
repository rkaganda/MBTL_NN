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
            if k_ in full_data.keys():
                full_data[k_] = full_data[k_] + i_
        full_data['next_state'] = full_data['state'][1:]
        full_data['next_state'].append([0.0] * len(full_data['next_state'][0]))

    return full_data


class RollingDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards, next_states, done, window):
        self.data = []

        index = window - 1
        while index < len(done) - window:
            self.data.append([
                torch.Tensor(states[index: index + window]),
                torch.Tensor(actions[index:index + window]),
                torch.Tensor(rewards[index:index + window]),
                torch.Tensor(next_states[index: index + window]),
                torch.Tensor(done[index: index + window]),
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


def train(model, target, optim, model_type, criterion, data, gamma):
    state = data[0].to(device)
    action = data[1].to(device)
    reward = data[2].to(device)
    next_state = data[3].to(device)
    done = data[4].to(device)

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
            print("next_q_values.size()={}".format(next_q_values.size()))
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

    reward_error = (q_values - q_targets).abs().mean()

    loss = criterion(q_values, q_targets)

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
    optim.step()

    return loss, reward_error, optim.param_groups[0]['lr']


def train_model(reward_paths, stats_path, model, target, optim, model_type, epochs, episode_num, window_size, gamma):
    writer = SummaryWriter(stats_path)
    reward_data = load_reward_data(reward_paths)

    criterion = nn.SmoothL1Loss()
    criterion.to(device)
    model.to(device)

    dataset = create_dataset(reward_data, window_size)
    sampler = torch.utils.data.RandomSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=len(dataset))

    eps_loss = 0.0
    total_reward_error = 0.0
    for epoch in tqdm(range(epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, reward_error, lr = train(model, target, optim, model_type, criterion, batch_data, gamma)
            eps_loss = eps_loss + train_loss
            total_reward_error = reward_error + total_reward_error
            # if step >= config.settings['batch_size']:
            #     break
    writer.add_scalar("Reward Error/train", total_reward_error/epochs, episode_num)
    writer.add_scalar("Loss/train", eps_loss/epochs, episode_num)
    writer.flush()
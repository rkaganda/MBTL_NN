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
    return full_data


class RollingDataset(torch.utils.data.Dataset):
    def __init__(self, states, actions, rewards, next_states, done, window):
        self.states = torch.Tensor(states).to(device)
        self.actions = torch.Tensor(actions).to(device)
        self.rewards = torch.Tensor(rewards).to(device)
        self.next_states = torch.Tensor(next_states).to(device)
        self.done = torch.Tensor(done).to(device)
        self.window = window

    def __getitem__(self, index):
        return [self.states[index:index+self.window],
                self.actions[index:index+self.window],
                self.rewards[index:index+self.window],
                self.next_states[index:index+self.window],
                self.done[index:index+self.window]]

    def __len__(self):
        return len(self.states) - self.window


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
    state = Variable(data[0])
    action = Variable(data[1])
    reward = Variable(data[2])
    next_state = Variable(data[3])
    done = Variable(data[4])

    model.train()
    train_loss = 0

    with torch.no_grad():
        next_q_values, _ = target(next_state)
        next_q_values = next_q_values.max(dim=1)[0]
        q_targets = reward + (1 - done) * next_q_values

    q_values, _ = model(state)
    q_values = q_values.gather(1, action[0].to(torch.int64)).squeeze(1)

    loss = F.smooth_l1_loss(q_values, q_targets[0])
    optim.zero_grad()
    loss.backward()
    optim.step()

    train_loss += loss.item()

    return train_loss, optim.param_groups[0]['lr']


def train_model(reward_paths, stats_path, model, target, optim, epochs, episode_num, window_size):
    writer = SummaryWriter(stats_path)
    reward_data = load_reward_data(reward_paths)

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)

    dataset = create_dataset(reward_data, window_size)
    sampler = torch.utils.data.RandomSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1)

    stats = {}

    for epoch in tqdm(range(0, epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, lr = train(model, target, optim, batch_data)
            writer.add_scalar("Loss/train", train_loss, step)
            writer.flush()
            stats[step] = {
                "epoch": epoch,
                "batch_size": len(batch_data[0]),
                "loss": train_loss,
                "learning rate": lr
            }


import copy
import logging

import torch
import torch.nn as nn
import torch.utils.data as td
from torch.autograd import Variable
import torch.nn.functional as F

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
    def __init__(self, states, actions, rewards, window):
        self.states = torch.Tensor(states).to(device)
        self.actions = torch.Tensor(actions).to(device)
        self.rewards = torch.Tensor(rewards).to(device)
        self.window = window

    def __getitem__(self, index):
        return [self.states[index:index+self.window],
                self.actions[index:index+self.window],
                self.rewards[index:index+self.window]]

    def __len__(self):
        return len(self.states) - self.window


def create_dataset(data, window_size):
    state_tensor = data['state']
    action_tensor = data['action']
    reward_tensor = data['reward']
    dataset = RollingDataset(
        states=state_tensor,
        actions=action_tensor,
        rewards=reward_tensor,
        window=window_size)

    return dataset


def train(model, target, optim, data):
    state = Variable(data[0])
    actions = Variable(data[1])
    rewards = Variable(data[2])

    model.train()
    train_loss = 0

    pred_q, _ = model(state)
    better_q = pred_q.clone().scatter_(1, actions[0].to(torch.int64), rewards[0].unsqueeze(1))

    # loss = F.smooth_l1_loss(pred_q, better_q).to(device)
    criteron = nn.CrossEntropyLoss()
    loss = criteron(pred_q, better_q.softmax(dim=1)).to(device)

    optim.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
        param.grad.to(device)
    optim.step()

    train_loss += loss.item()

    return train_loss, optim.param_groups[0]['lr']


def train_model(reward_paths, stats_path, model, target, optim, epochs, episode_num, window_size):
    reward_data = load_reward_data(reward_paths)
    batch_size = config.settings['batch_size']

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion.to(device)
    model.to(device)

    dataset = create_dataset(reward_data, window_size)
    sampler = torch.utils.data.SequentialSampler(dataset)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1)

    stats = {}

    for epoch in tqdm(range(0, epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, lr = train(model, target, optim, batch_data)
            stats[step] = {
                "epoch": epoch,
                "batch_size": len(batch_data[0]),
                "loss": train_loss,
                "learning rate": lr
            }

    with open("{}/{}.json".format(stats_path, episode_num), 'a') as f_writer:
        f_writer.write(json.dumps(stats))

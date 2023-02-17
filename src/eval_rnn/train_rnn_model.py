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
        state_size = len(full_data['state'][0][0])
        sequence_size = len(full_data['state'][0])
        next_state = copy.deepcopy(file_dict['state'])
        next_state = next_state[1:]
        next_state.append([list(0 for n in range(state_size)) for n in range(sequence_size)])
        full_data['next_state'] = next_state
        done = [0] * len(next_state)
        done[-1] = 1
        full_data['done'] = done
    return full_data


def create_dataset(data):
    state_tensor = torch.Tensor(data['state']).to(device)
    action_tensor = torch.Tensor(data['action']).to(device)
    reward_tensor = torch.Tensor(data['reward']).to(device)
    next_state_tensor = torch.Tensor(data['next_state']).to(device)
    done = torch.Tensor(data['done']).to(device)

    return td.TensorDataset(state_tensor, action_tensor, reward_tensor, next_state_tensor, done)


def train(model, target, optim, data):
    state = Variable(data[0])
    next_states = Variable(data[3])
    actions = Variable(data[1])
    rewards = Variable(data[2])
    done = Variable(data[4])

    batch_size = len(data[0])

    model.train()
    train_loss = 0

    pred_q, _ = model(state)
    pred_q = pred_q.gather(1, actions.type(torch.int64))
    next_state_q_vals = torch.zeros(batch_size).to(device)

    for idx, next_state in enumerate(next_states):
        if done[idx] == 1:
            next_state_q_vals[idx] = -1
        else:
            next_state_q, hidden_state = target(next_state.unsqueeze(0))
            next_state_q_vals[idx] = next_state_q[-1].argmax()

    better_pred = (rewards + next_state_q_vals).unsqueeze(1)

    loss = F.smooth_l1_loss(pred_q, better_pred).to(device)
    optim.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
        param.grad.to(device)
    optim.step()

    train_loss += loss.item()

    return train_loss, optim.param_groups[0]['lr']


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

    for epoch in tqdm(range(0, epochs)):
        for step, batch_data in enumerate(train_loader):
            train_loss, lr = train(model, target, optim, batch_data)
            stats[step] = {
                "epoch": epoch,
                "batch_size": len(batch_data[0]),
                "loss": train_loss,
                "learning rate": lr
            }
            break

    with open("{}/{}.json".format(stats_path, episode_num), 'a') as f_writer:
        f_writer.write(json.dumps(stats))

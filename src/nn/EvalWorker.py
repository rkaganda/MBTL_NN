import time
import logging
import traceback
import datetime
import math
import random

import multiprocessing as mp
import numpy as np

import torch
import torch.nn as nn
# import torch.multiprocessing as mp

import mbtl_input
import melty_state
import nn.model as model
import nn.eval_util as eval_util
import config
import calc_reward
import train_dqn_model

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class EvalWorker(mp.Process):
    def __init__(self, game_states, frames_per_evaluation, reaction_delay, env_status, eval_status,
                 input_index, input_index_max, state_format,
                 learning_rate, player_idx, frame_list, neutral_index):
        super(EvalWorker, self).__init__()
        self.states = game_states
        self.frames_per_evaluation = frames_per_evaluation
        self.reaction_delay = reaction_delay
        self.env_status = env_status
        self.eval_status = eval_status
        self.input_index = input_index
        self.input_index_max = input_index_max
        self.learning_rate = learning_rate
        self.player_idx = player_idx
        self.frame_list = frame_list

        self.state_format = state_format
        self.neutral_index = neutral_index

        self.model = None
        self.optimizer = None

        self.target = None

        self.episode_number = 0

        self.run_count = 1
        self.epsilon = 1

    # normalize the state attributes between 0, 1 using precalculated min max
    def normalize_state(self, state):
        norm_state = dict()

        # normalize game state
        minmax = self.state_format['minmax']
        game_state = melty_state.calc_extra_states(state['game'])
        norm_state['game'] = list()
        for p_idx in [0, 1]:  # for each player state
            for attrib in self.state_format['attrib']:  # for each attribute
                # if value is outside min max
                if game_state[p_idx][attrib] > minmax[attrib]['max'] or \
                        game_state[p_idx][attrib] < minmax[attrib]['min']:
                    pass
                    # logging.info("state out of bounds: attrib={}, state={}, minmax={}".format(
                    #     attrib, game_state[player_idx][attrib], minmax[attrib]))
                # range = max - min
                min_max_diff = minmax[attrib]['max'] - minmax[attrib]['min']

                # norm = (value - min) / (max - min) unless max - min = 0
                norm_state['game'].append((game_state[p_idx][attrib] - minmax[attrib]['min']) / (
                    min_max_diff) if min_max_diff != 0 else 0)

        # normalize input
        input_index = state['input'][self.player_idx]
        # print("input_index = {}".format(input_index))
        norm_state['input'] = input_index / (self.input_index_max + 1)

        return norm_state

    def setup_model(self):
        self.model, self.optimizer = model.setup_model(
            frames_per_observation=self.frames_per_evaluation,
            input_state_size=self.input_index_max + 1,
            state_state_size=len(self.state_format['attrib']),
            learning_rate=self.learning_rate
        )

        self.target, _ = model.setup_model(
            frames_per_observation=self.frames_per_evaluation,
            input_state_size=self.input_index_max + 1,
            state_state_size=len(self.state_format['attrib']),
            learning_rate=self.learning_rate
        )
        self.episode_number, self.run_count = eval_util.get_next_episode(player_idx=self.player_idx)

        if self.episode_number > 0:
            print("resuming player:{} on eps:{} run_count:{}".format(self.player_idx, self.episode_number,
                                                                     self.run_count))
            model.load_model(self.model, self.optimizer, self.player_idx, self.episode_number, device)
            print("loaded model")
        else:
            print("fresh model")
            torch.manual_seed(0)

            model.weights_init_uniform_rule(model)

            model.save_model(self.model, self.optimizer, self.player_idx, episode_num=-1)

        self.target.load_state_dict(self.model.state_dict())
        self.model = self.model.to(device)
        self.target = self.target.to(device)

        self.target.eval()

        # warm up model
        with torch.no_grad():
            in_tensor = torch.ones(
                self.frames_per_evaluation * (1 + (len(self.state_format['attrib']) * 2))).to(device)
            out_tensor = self.model(in_tensor)

    def round_cleanup(self, normalized_states, model_output):
        eval_util.store_eval_output(
            normalized_states,
            self.states,
            model_output,
            self.state_format,
            self.player_idx,
            self.episode_number,
        )
        if self.run_count % config.settings['tau'] == 0:
            print("loading target from model")
            self.target.load_state_dict(self.model.state_dict())

        if config.settings['save_model'] and (self.run_count % config.settings['count_save']) == 0:
            print("epoch cleanup...")
            self.reward_train()
            model.save_model(self.model, self.optimizer, self.player_idx, episode_num=self.episode_number)
            self.episode_number += 1

    def reward_train(self):
        reward_path = "data/eval/{}/reward/{}/{}".format(config.settings['run_name'], self.player_idx,
                                                         self.episode_number)
        eval_path = "data/eval/{}/evals/{}/{}".format(config.settings['run_name'], self.player_idx, self.episode_number)
        stats_path = "data/eval/{}/stats/{}".format(config.settings['run_name'], self.player_idx)
        reward_paths = calc_reward.generate_rewards(
            reward_path=reward_path,
            eval_path=eval_path,
            reward_columns=config.settings['reward_columns'][self.player_idx],
            falloff=config.settings['reward_falloff'],
            player_idx=self.player_idx,
            reaction_delay=config.settings['reaction_delay'],
            hit_preframes=config.settings['hit_preframes'],
            reward_gamma=config.settings['reward_gamma']
        )
        if not config.settings['last_episode_only']:
            reward_paths = eval_util.get_reward_paths(self.player_idx)
        train_dqn_model.train_model(reward_paths, stats_path, self.model, self.target, self.optimizer,
                                    config.settings['epochs'],
                                    self.episode_number)

    def run(self):
        try:
            self.setup_model()
            did_store = False

            normalized_states = list()
            model_output = dict()
            last_normalized_index = 0
            last_evaluated_index = 0

            # RNG
            final_epsilon = config.settings['final_epsilon']
            initial_epsilon = config.settings['initial_epsilon']
            epsilon_decay = config.settings['epsilon_decay']

            self.eval_status['eval_ready'] = True
            while not self.eval_status['kill_eval']:
                self.eval_status['eval_ready'] = True
                while not self.env_status['round_done'] and not self.eval_status['kill_eval']:
                    did_store = False
                    if len(self.states) > len(normalized_states):  # if there are game states to normalize
                        # normalize a frame and append to to normalized states
                        normalized_states.append(
                            self.normalize_state(self.states[last_normalized_index])
                        )
                        last_normalized_index = last_normalized_index + 1

                        # if reaction time has passed, and we have enough frames to eval
                        if ((last_normalized_index - self.reaction_delay) >= last_evaluated_index) and (
                                (len(normalized_states) - self.reaction_delay) >= self.frames_per_evaluation):

                            # exploration calculation
                            esp_count = self.run_count

                            eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                                            math.exp(-1. * esp_count / epsilon_decay)

                            self.epsilon = eps_threshold

                            # create slice for evaluation
                            evaluation_frames = normalized_states[:-self.reaction_delay]
                            evaluation_frames = evaluation_frames[-self.frames_per_evaluation:]

                            # flatten for input into model
                            flat_frames = []
                            for f_ in evaluation_frames:
                                flat_frames = flat_frames + [f_['input']] + f_['game']

                            if random.random() < eps_threshold:
                                detached_out = torch.Tensor(np.random.rand(self.input_index_max + 1))
                            else:
                                # create tensor
                                in_tensor = torch.Tensor(flat_frames).to(device)

                                # input data into model
                                with torch.no_grad():
                                    out_tensor = self.model(in_tensor)

                                detached_out = out_tensor.detach().cpu()
                            try:
                                if config.settings['use_best_action']:
                                    action_index = torch.argmax(detached_out).numpy()
                                else:
                                    r = torch.clone(detached_out)
                                    r = r + torch.abs(torch.min(r))
                                    r = r / torch.sum(r)

                                    action_index = np.random.choice(r.size(0), 1, p=r.numpy())[0]
                            except RuntimeError as e:
                                print("in_tensor={}".format(in_tensor))
                                print("detached_out={}".format(action_index))
                                raise e

                            self.input_index.value = action_index

                            # store model output
                            model_output[last_evaluated_index] = {
                                'output': list(detached_out.numpy()),
                                'frame': self.frame_list[-1],
                                # 'state': flat_frames,
                                'states': len(self.states),
                                'norm_states': len(normalized_states),
                                'last_evaluated_index': last_evaluated_index,
                                'last_normalized_index': last_normalized_index,
                                'window': [
                                    last_normalized_index - self.reaction_delay - self.frames_per_evaluation,
                                    last_normalized_index - self.reaction_delay - 1
                                ],
                                "epsilon": self.epsilon
                                # 'input_tensor': flat_frames
                            }

                            # increment evaluated index
                            last_evaluated_index = last_evaluated_index + 1
                    else:
                        pass  # no states yet
                if not did_store and len(normalized_states) > 0:
                    print("eps_threshold={}".format(self.epsilon))
                    logger.debug("{} eval cleanup".format(self.player_idx))
                    self.eval_status['eval_ready'] = False
                    logger.debug("{} stopping eval".format(self.player_idx))
                    self.round_cleanup(normalized_states, model_output)
                    did_store = True
                    del normalized_states[:]
                    model_output.clear()
                    last_normalized_index = 0
                    last_evaluated_index = 0
                    self.run_count = self.run_count + 1
                    logger.debug("{} finished cleanup".format(self.player_idx))
                    self.eval_status['storing_eval'] = False  # finished storing eval
        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
            raise identifier

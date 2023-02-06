import logging
import traceback
import math
import random
import copy

import multiprocessing as mp
import numpy as np

import torch
# import torch.multiprocessing as mp

import melty_state
import eval.model as model
import eval.eval_util as eval_util
import config
import calc_reward
from eval import train_dqn_model

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class EvalWorker(mp.Process):
    def __init__(self, game_states: list, frames_per_evaluation: int, reaction_delay: int, env_status: dict,
                 eval_status: dict,
                 input_index: int, input_index_max: int, state_format: dict, player_facing_flag: int,
                 learning_rate: float, player_idx: int, frame_list: list, neutral_action_index: int, input_lookback: int):
        """

        :param game_states: list of game states for the current round index by frame
        :param frames_per_evaluation: num of state frames used for model input each frame
        :param reaction_delay: num of frames to delay eval
        :param env_status:
        :param eval_status: status of this eval
        :param input_index: current input index action
        :param input_index_max: max input index
        :param state_format: min, max values of each attrib used for norm
        :param learning_rate: learning rate of the model
        :param player_idx: index of player for this eval
        :param frame_list: list of frame numbers
        :param neutral_action_index: index for neutral action (no buttons pressed)
        :param input_lookback: how many input frames add to model input
        :param input_lookback: the flag for what direction the player is facing
        """
        super(EvalWorker, self).__init__()
        self.states = game_states
        self.updated_states = [[], []]
        self.frames_per_evaluation = frames_per_evaluation
        self.reaction_delay = reaction_delay
        self.env_status = env_status
        self.eval_status = eval_status
        self.input_index = input_index
        self.input_index_max = input_index_max
        self.learning_rate = learning_rate
        self.player_idx = player_idx
        self.frame_list = frame_list
        self.input_lookback = input_lookback
        self.player_facing_flag = player_facing_flag

        self.relative_states = []
        self.state_format = state_format
        self.neutral_action_index = neutral_action_index
        self.norm_neut_index = neutral_action_index / (input_index_max - 1)

        self.model = None
        self.target = None
        self.optimizer = None

        self.episode_number = 0

        self.run_count = 1
        self.epsilon = 1
        self.reward_paths = []

    def normalize_state(self, state: dict) -> (dict, dict):
        """
        normalize the state attributes between 0, 1 using precalculated min max
        :param state:
        :return:
        """
        norm_state = dict()

        # normalize game state
        minmax = self.state_format['minmax']
        game_state, player_facing_flag = melty_state.encode_relative_states(copy.deepcopy(state['game']), self.player_idx)
        norm_state['game'] = list()
        for p_idx in [0, 1]:  # for each player state
            for attrib in self.state_format['attrib']:  # for each attribute
                if attrib not in game_state[p_idx]:
                    print("failed attrib={}".format(attrib))
                    print(game_state[p_idx])
                # if value is outside min max
                if game_state[p_idx][attrib] > minmax[attrib]['max'] or \
                        game_state[p_idx][attrib] < minmax[attrib]['min']:
                    pass
                # range = max - min
                min_max_diff = minmax[attrib]['max'] - minmax[attrib]['min']

                # norm = (value - min) / (max - min) unless max - min = 0
                norm_state['game'].append((game_state[p_idx][attrib] - minmax[attrib]['min']) / (
                    min_max_diff) if min_max_diff != 0 else 0)

        # normalize input
        input_index = state['input'][self.player_idx]
        norm_state['input'] = input_index / (self.input_index_max + 1)

        return norm_state, {'game': game_state, 'input': state['input']}, player_facing_flag

    def setup_model(self):
        """
        init model, target, optim
        load existing if exists for run
        :return:
        """
        self.model, self.optimizer = model.setup_model(
            frames_per_observation=self.frames_per_evaluation,
            input_lookback=self.input_lookback,
            input_state_size=self.input_index_max + 1,
            state_state_size=len(self.state_format['attrib']),
            learning_rate=self.learning_rate
        )

        self.target, _ = model.setup_model(
            frames_per_observation=self.frames_per_evaluation,
            input_lookback=self.input_lookback,
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

        if not config.settings['last_episode_only']:
            self.reward_paths = eval_util.get_reward_paths(self.player_idx)

        self.target.eval()

        # warm up model
        with torch.no_grad():
            in_tensor = torch.ones(
                (self.frames_per_evaluation * (len(self.state_format['attrib']) * 2)) + self.input_lookback).to(device)
            out_tensor = self.model(in_tensor)

    def round_cleanup(self, normalized_states, model_output):
        """
        store for this round
        train and update target net if necessary
        :param normalized_states:
        :param model_output:
        :return:
        """
        eval_util.store_eval_output(
            normalized_states,
            self.relative_states,
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
        reward_path = "{}/eval/{}/reward/{}/{}".format(config.settings['data_path'], config.settings['run_name'],
                                                       self.player_idx,
                                                       self.episode_number)
        eval_path = "{}/eval/{}/evals/{}/{}".format(config.settings['data_path'], config.settings['run_name'],
                                                    self.player_idx, self.episode_number)
        stats_path = "{}/eval/{}/stats/{}".format(config.settings['data_path'], config.settings['run_name'],
                                                  self.player_idx)
        reward_paths = calc_reward.generate_rewards(
            reward_path=reward_path,
            eval_path=eval_path,
            reward_columns=config.settings['reward_columns'][self.player_idx],
            falloff=config.settings['reward_falloff'],
            player_idx=self.player_idx,
            reaction_delay=config.settings['reaction_delay'],
            hit_preframes=config.settings['hit_preframes'],
            atk_preframes=config.settings['atk_preframes'],
            whiff_reward=config.settings['whiff_reward'],
            reward_gamma=config.settings['reward_gamma']
        )
        if config.settings['last_episode_only']:
            reward_sample = [reward_paths]
        else:
            self.reward_paths.append(reward_paths)
            if len(self.reward_paths) > config.settings['episode_sample_size']:
                reward_sample = random.sample(self.reward_paths, config.settings['episode_sample_size'])
            else:
                reward_sample = self.reward_paths

        train_dqn_model.train_model(reward_sample, stats_path, self.model, self.target, self.optimizer,
                                    config.settings['epochs'],
                                    self.episode_number)

    def run(self):
        try:
            self.setup_model()
            did_store = False

            normalized_states = list()
            normalized_inputs = list()
            model_output = dict()
            last_normalized_index = 0
            last_evaluated_index = 0

            # dora please
            final_epsilon = config.settings['final_epsilon']
            initial_epsilon = config.settings['initial_epsilon']
            epsilon_decay = config.settings['epsilon_decay']

            self.eval_status['eval_ready'] = True  # eval is ready
            while not self.eval_status['kill_eval']:
                self.eval_status['eval_ready'] = True  # still ready
                # while round is live and no kill event
                while not self.env_status['round_done'] and not self.eval_status['kill_eval']:
                    did_store = False  # didn't store for this round yet
                    if len(self.states) > len(normalized_states):  # if there are frames to normalize
                        # normalize a frame and append to to normalized states
                        normalized_state, relative_state, player_facing_flag = \
                            self.normalize_state(self.states[last_normalized_index])

                        if last_normalized_index != len(self.relative_states):
                            raise IndexError
                        self.relative_states.append(relative_state)

                        normalized_states.append(
                            normalized_state
                        )
                        normalized_inputs.append(normalized_states[-1]['input'])
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

                            input_len = len(normalized_inputs)
                            if input_len < self.input_lookback:  # if not enough inputs pad with neutral input
                                input_frames = \
                                    ([self.norm_neut_index] * (self.input_lookback - input_len)) + normalized_inputs
                            else:
                                input_frames = normalized_inputs[-self.input_lookback:]

                            # flatten for input into model
                            flat_frames = []
                            for f_ in evaluation_frames:
                                flat_frames = flat_frames + f_['game']
                            flat_frames = input_frames + flat_frames  # put input state in front of game state

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
                                action_index = torch.argmax(detached_out).numpy()
                                action_index = self.neutral_action_index
                            except RuntimeError as e:
                                logger.debug("in_tensor={}".format(in_tensor))
                                logger.debug("detached_out={}".format(action_index))
                                logger.exception(e)
                                raise e

                            self.input_index.value = action_index
                            self.player_facing_flag.value = player_facing_flag

                            # store model output
                            normalized_states[last_evaluated_index]['input'] = input_frames
                            model_output[last_evaluated_index] = {
                                'output': list(detached_out.numpy()),
                                'frame': self.frame_list[-1],
                                'state': flat_frames,
                                'states': len(self.states),
                                'norm_states': len(normalized_states),
                                'last_evaluated_index': last_evaluated_index,
                                'last_normalized_index': last_normalized_index,
                                'window': [
                                    last_normalized_index - self.reaction_delay - self.frames_per_evaluation,
                                    last_normalized_index - self.reaction_delay - 1
                                ],
                                "epsilon": self.epsilon
                            }

                            # increment evaluated index
                            last_evaluated_index = last_evaluated_index + 1
                    else:
                        pass  # no states yet
                if not did_store and len(model_output) > 0:  # if we didn't store yet and there are states to store
                    print("eps_threshold={}".format(self.epsilon))
                    logger.debug("{} eval cleanup".format(self.player_idx))
                    self.eval_status['eval_ready'] = False  # eval is not ready
                    logger.debug("{} stopping eval".format(self.player_idx))
                    normalized_states = normalized_states[0:last_evaluated_index]  # trim states not used
                    self.round_cleanup(normalized_states, model_output)  # train, store
                    did_store = True
                    del normalized_states[:]  # clear norm state for round
                    model_output.clear()  # clear
                    last_normalized_index = 0
                    last_evaluated_index = 0
                    self.run_count = self.run_count + 1
                    logger.debug("{} finished cleanup".format(self.player_idx))
                    self.eval_status['storing_eval'] = False  # finished storing eval
        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
            raise identifier

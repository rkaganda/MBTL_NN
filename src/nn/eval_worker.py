import logging
import traceback
import math
import random
import copy
import numpy as np

import multiprocessing as mp
from multiprocessing import Value

import torch
from torch.utils.tensorboard import SummaryWriter

import melty_state
import nn.model_util as model_util
import nn.eval_util as eval_util
import nn.train_util as train_util
import nn.rnn_model as rnn_model
import nn.transformer_model as transformer_model
import config
from nn import calc_reward
from action_scripts import action_script

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


class EvalWorker(mp.Process):
    def __init__(self, game_states: list, frames_per_evaluation: int, reaction_delay: int, env_status: dict,
                 eval_status: dict,
                 input_index_max: int, state_format: dict, player_facing_flags: dict,
                 learning_rate: float, player_idx: int, frame_list: list, neutral_action_index: int,
                 action_buffer: dict, current_state_frame: Value
                 ):
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
        :param player_facing_flag: the flag for what direction the player is facing
        """
        super(EvalWorker, self).__init__()
        self.states = game_states
        self.updated_states = [[], []]
        self.frames_per_evaluation = frames_per_evaluation
        self.reaction_delay = reaction_delay
        self.env_status = env_status
        self.eval_status = eval_status
        self.input_index_max = input_index_max
        self.learning_rate = learning_rate
        self.player_idx = player_idx
        self.frame_list = frame_list
        self.player_facing_flags = player_facing_flags
        self.action_buffer = action_buffer
        self.current_state_frame = current_state_frame

        # dora please

        self.relative_states = []
        self.state_format = state_format
        self.neutral_action_index = neutral_action_index
        self.norm_neut_index = neutral_action_index / (input_index_max - 1)
        self.mean_pred_q = 0
        self.mean_pred_explore_count = 0

        self.model = None
        self.target = None
        self.optimizer = None
        self.model_config = config.settings['p{}_model'.format(self.player_idx)]

        # get total size of categorical features
        one_hot_size = 0
        for c, c_list in self.state_format['categorical'].items():
            if c in self.model_config['state_features']:
                one_hot_size = len(c_list) + one_hot_size
        # sum features, value features + categorical features + last input
        value_size = 0
        for v in self.state_format['values']:
            if v in self.model_config['state_features']:
                value_size = value_size + 1
        self.model_input_size = (value_size*2) + (one_hot_size*2) + (input_index_max+1)
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
        game_state, player_facing_flag = melty_state.encode_relative_states(
            copy.deepcopy(state['game']), self.player_idx)
        norm_state['game'] = list()
        for p_idx in [0, 1]:  # for each player state
            for attrib in self.state_format['values']:  # for each attribute
                if attrib in self.model_config['state_features']:
                    if attrib not in game_state[p_idx]:
                        print("failed attrib={}".format(attrib))
                        print(game_state[p_idx])
                    min_max_diff = minmax[attrib]['max'] - minmax[attrib]['min']
                    # norm = (value - min) / (max - min) unless max - min = 0
                    norm_state['game'].append((game_state[p_idx][attrib] - minmax[attrib]['min']) / (
                        min_max_diff) if min_max_diff != 0 else 0)

            # encode categorical
            categorical_states = []
            for category in self.state_format['categories']:
                if category in self.model_config['state_features']:
                    category_list = [0] * len(self.state_format['categorical'][category])
                    category_list[
                        self.state_format['categorical'][category][str(game_state[p_idx][category])]] = 1  # encode
                    categorical_states = categorical_states + category_list
            # append categorical
            norm_state['game'] = norm_state['game'] + categorical_states

        # encode input
        input_index = state['input'][self.player_idx]
        norm_state['input'] = [0]*(self.input_index_max + 1)  # create list of size
        norm_state['input'][input_index] = 1  # encode

        return norm_state, {'game': game_state, 'input': state['input']}, player_facing_flag

    def setup_model(self):
        """
        init model, target, optim
        load existing if exists for run
        :return:
        """
        torch.manual_seed(0)
        if self.model_config['type'] == 'rnn':
            model = rnn_model
        elif self.model_config['type'] == 'transformer':
            model = transformer_model
        else:
            raise Exception("{} type model invalid".format(self.model_config['type']))

        self.model, self.optimizer = model.setup_model(
            actions_size=self.input_index_max + 1,
            input_size=self.model_input_size,
            learning_rate=self.learning_rate,
            hyperparams=self.model_config['hyperparams']
        )

        self.target, _ = model.setup_model(
            actions_size=self.input_index_max + 1,
            input_size=self.model_input_size,
            learning_rate=self.learning_rate,
            hyperparams=self.model_config['hyperparams']
        )

        self.episode_number, self.run_count = eval_util.get_next_episode(player_idx=self.player_idx)
        if not model_util.load_model(self.model, self.optimizer, self.player_idx):
            print("fresh model")

        self.target.load_state_dict(self.model.state_dict())
        self.model = self.model.to(device)
        print("p{} model: \n{}".format(self.player_idx, self.model))
        self.target = self.target.to(device)

        if not config.settings['last_episode_only']:
            self.reward_paths = eval_util.get_reward_paths(self.player_idx)

        self.target.eval()

        # warm up model
        in_tensor = torch.rand(1, self.frames_per_evaluation, self.model_input_size+self.model.input_padding).to(device)

        # input data into model
        with torch.no_grad():
            _ = self.model(in_tensor)

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
            self.reward_train()
            model_util.save_model(self.model, self.optimizer, self.player_idx)
            self.episode_number += 1

    def reward_train(self):
        reward_path = "{}/eval/{}/reward/{}/{}".format(config.settings['data_path'], config.settings['run_name'],
                                                       self.player_idx,
                                                       self.episode_number)
        eval_path = "{}/eval/{}/evals/{}/{}".format(config.settings['data_path'], config.settings['run_name'],
                                                    self.player_idx, self.episode_number)
        stats_path = "{}/runs/{}_{}".format(config.settings['data_path'], config.settings['run_name'],
                                            self.player_idx)

        writer = SummaryWriter(stats_path)
        writer.add_scalar("{}/{}".format("explore", "mean_pred_q"), self.mean_pred_q, self.episode_number)
        writer.add_scalar("{}/{}".format("explore", "explore_count"), self.mean_pred_explore_count, self.episode_number)

        try:
            reward_paths = calc_reward.generate_rewards(
                reward_path=reward_path,
                eval_path=eval_path,
                reward_columns=self.model_config['reward_columns'][0],
                falloff=self.model_config['reward_falloff'],
                player_idx=self.player_idx,
                reaction_delay=self.reaction_delay,
                atk_preframes=self.model_config['atk_preframes'],
                whiff_reward=self.model_config['whiff_reward'],
                reward_gamma=self.model_config['reward_gamma'],
                frames_per_observation=self.model_config['frames_per_observation'],
                stats_path=stats_path,
                episode_number=self.episode_number,
                full_reward=self.model_config['full_reward'],
            )
            if config.settings['last_episode_only']:
                reward_sample = [reward_paths]
            else:
                self.reward_paths.append(reward_paths)
                if len(self.reward_paths) > config.settings['episode_sample_size']:
                    reward_sample = random.sample(self.reward_paths, config.settings['episode_sample_size'])
                else:
                    reward_sample = self.reward_paths

            train_util.train_model(reward_sample, stats_path, self.model, self.target, self.optimizer,
                                                self.model_config['type'],
                                                config.settings['epochs'],
                                                self.episode_number, self.frames_per_evaluation,
                                                self.model_config['reward_gamma'])
        except calc_reward.ZeroRewardDiff:
            print("Zero reward in eval={}".format(eval_path))
            logger.debug("Zero reward in eval={}".format(eval_path))

    def run(self):
        try:
            self.setup_model()
            did_store = False

            normalized_states = list()
            normalized_inputs = list()
            player_facing_flags = list()
            model_output = dict()
            last_normalized_index = 0
            last_evaluated_index = 0

            # dora please
            final_epsilon = config.settings['final_epsilon']
            initial_epsilon = config.settings['initial_epsilon']
            epsilon_decay = config.settings['epsilon_decay']
            eps_threshold = initial_epsilon
            esp_count = 0
            no_explore_count = 0
            explore_better_action = config.settings['probability_action']
            self.mean_pred_q = 0
            self.mean_pred_explore_count = 0

            # TODO ACTION SCRIPT
            act_script = action_script.ActionScript()

            self.eval_status['eval_ready'] = True  # eval is ready
            print("eps={} no_explore={}".format(self.epsilon, no_explore_count))
            while not self.eval_status['kill_eval']:
                self.eval_status['eval_ready'] = True  # still ready
                # while round is live and no kill event
                while not self.env_status['round_done'] and not self.eval_status['kill_eval']:
                    did_store = False  # didn't store for this round yet
                    last_state_index = len(self.states) - 1
                    if last_normalized_index < last_state_index:
                        # normalize a frame and append to to normalized states
                        normalized_state, relative_state, player_facing_flag = \
                            self.normalize_state(self.states[last_normalized_index])

                        if last_normalized_index != len(self.relative_states):
                            raise IndexError
                        self.relative_states.append(relative_state)

                        normalized_states.append(
                            normalized_state
                        )
                        player_facing_flags.append(
                            player_facing_flag
                        )
                        normalized_inputs.append(normalized_states[-1]['input'])
                        last_normalized_index = last_normalized_index + 1

                    # if reaction time has passed, and we have enough frames to eval
                    if last_normalized_index > last_evaluated_index:
                        # create slice for evaluation
                        normalized_states = normalized_states

                        # flatten for input into model
                        flat_frames = []

                        for f_idx in range(last_evaluated_index - self.frames_per_evaluation+1,
                                           last_evaluated_index+1):
                            if f_idx < 0:
                                flat_frames.append(
                                    [0.0]*len(normalized_states[0]['game']) +
                                    [0.0]*len(normalized_inputs[0]) +
                                    [0.0]*self.model.input_padding
                                )
                            else:
                                flat_frames.append(
                                    normalized_states[f_idx]['game'] +
                                    normalized_inputs[f_idx] +
                                    [0.0]*self.model.input_padding
                                )
                        # # TODO ACTION
                        # if self.player_idx in [1]:
                        #     action_index = act_script.get_action(self.states, last_evaluated_index)
                        #
                        # if act_script.get_current_frame() != -1:
                        #     max_q = 0
                        #     detached_out = torch.zeros(self.input_index_max + 1)

                        # create tensor
                        in_tensor = torch.Tensor(flat_frames).to(device)

                        # input data into model

                        self.model.eval()
                        with torch.no_grad():
                            if self.model_config['type'] == 'rnn':
                                # rnn returns last out and hidden state
                                out_tensor, _ = self.model(in_tensor.unsqueeze(0))
                            elif self.model_config['type'] == 'transformer':
                                # transformer returns full sequence
                                in_tensor = in_tensor.unsqueeze(0).transpose(0, 1)  # reshape to (seq_length, batch_size, features)
                                out_tensor = self.model(in_tensor)[-1, :, :]

                        detached_out = out_tensor[-1].detach().cpu()

                        explore = 0
                        if random.random() < eps_threshold:  # explore
                            explore = 1
                            # if no input mask
                            if self.model_config['input_mask'] is None:
                                action_index = random.randrange(0, len(detached_out))  # select random action
                            else:
                                # select random from mask
                                action_index = random.choice(
                                    self.model_config['input_mask'])
                        else:  # no explore
                            # no mask
                            if self.model_config['input_mask'] is None:
                                action_index = torch.argmax(detached_out).numpy().item()  # max predicted Q
                            else:
                                # max predicted Q with mask
                                action_index = torch.argmax(
                                    detached_out[self.model_config['input_mask']]
                                ).numpy().item()
                        max_q = detached_out[action_index].numpy().item()  # store predicted Q

                        if explore_better_action and max_q < self.mean_pred_q:  # explore better action
                            explore = 2
                            self.mean_pred_explore_count = self.mean_pred_explore_count + 1
                            out_clone = detached_out.clone()  # clone the model predicted Qs
                            if out_clone.min() < 0:  # normalize so that min is 0
                                out_clone = out_clone - out_clone.min()

                            # no mask
                            if self.model_config['input_mask'] is None:
                                # select action using predicted Q as probability distribution
                                action_index = torch.multinomial(out_clone, 1).numpy().item()
                            else:
                                # select action using predicted Q as probability distribution from input mask
                                action_index = torch.multinomial(
                                    out_clone[self.model_config['input_mask']],
                                    1).numpy().item()

                        self.mean_pred_q = self.mean_pred_q + max_q
                        self.mean_pred_q = self.mean_pred_q / 2
                        try:
                            pass
                            # eval_util.print_q(
                            #     cur_frame=len(self.states)-1,
                            #     eval_frame=last_normalized_index,
                            #     action=action_index,
                            #     q=max_q,
                            #     mean_q=self.mean_pred_q
                            # )

                        except RuntimeError as e:
                            logger.debug("in_tensor={}".format(in_tensor))
                            logger.debug("detached_out={}".format(action_index))
                            logger.exception(e)
                            raise e

                        # self.input_index.value = action_index
                        self.action_buffer[
                            last_evaluated_index+self.model_config['reaction_delay']+1
                        ] = action_index
                        self.player_facing_flags[
                            last_evaluated_index+self.model_config['reaction_delay']+1
                        ] = player_facing_flags[last_evaluated_index]

                        # store model output
                        model_output[last_evaluated_index] = {
                            'pred_q': max_q,
                            'action_index': action_index,
                            'output': list(detached_out.numpy()),
                            'frame': self.frame_list[-1],
                            'input': flat_frames[-1],
                            'states': len(self.states),
                            'norm_states': len(normalized_states),
                            'last_evaluated_index': last_evaluated_index,
                            'last_normalized_index': last_normalized_index,
                            "epsilon": self.epsilon,
                            "explore": explore
                        }

                        # increment evaluated index
                        last_evaluated_index = last_evaluated_index + 1
                    else:
                        pass  # no states yet
                if not did_store and len(model_output) > 0:  # if we didn't store yet and there are states to store
                    # dora
                    eps_threshold = final_epsilon + (initial_epsilon - final_epsilon) * \
                                    math.exp(-1. * esp_count / epsilon_decay)

                    if eps_threshold <= config.settings['eps_explore_threshold']:
                        no_explore_count = no_explore_count + 1

                    if no_explore_count >= config.settings['explore_reset']:
                        print("explore_reset")
                        esp_count = 0
                        no_explore_count = 0
                        if config.settings['probability_action']:
                            explore_better_action = not explore_better_action
                            print("explore_better_action={}".format(explore_better_action))
                    else:
                        esp_count = esp_count + 1

                    self.epsilon = round(eps_threshold, 2)
                    logger.debug("{} eval cleanup".format(self.player_idx))
                    self.eval_status['eval_ready'] = False  # eval is not ready
                    logger.debug("{} stopping eval".format(self.player_idx))
                    normalized_states = normalized_states[0:last_evaluated_index]  # trim states not used
                    self.round_cleanup(normalized_states, model_output)  # train, store
                    did_store = True
                    del self.relative_states[:]  # clear norm state for round
                    del normalized_states[:]  # clear norm state for round
                    model_output.clear()  # clear
                    last_normalized_index = 0
                    last_evaluated_index = 0
                    self.mean_pred_q = 0
                    self.mean_pred_explore_count = 0
                    self.run_count = self.run_count + 1
                    logger.debug("{} finished cleanup".format(self.player_idx))
                    act_script.reset()
                    self.eval_status['storing_eval'] = False  # finished storing eval
                    print("eps={} no_explore={}".format(self.epsilon, no_explore_count))
        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
            raise identifier

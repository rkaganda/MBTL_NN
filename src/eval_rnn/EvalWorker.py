import logging
import traceback
import math
import random
import copy

import multiprocessing as mp
import numpy as np

import torch
# import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import melty_state
import eval_rnn.model as model
import eval_rnn.eval_util as eval_util
import config
from eval_rnn import calc_reward
from eval_rnn import train_rnn_model
from action_scripts import action_script

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class EvalWorker(mp.Process):
    def __init__(self, game_states: list, frames_per_evaluation: int, reaction_delay: int, env_status: dict,
                 eval_status: dict,
                 input_index: int, input_index_max: int, state_format: dict, player_facing_flag: int,
                 learning_rate: float, player_idx: int, frame_list: list, neutral_action_index: int):
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
        self.input_index = input_index
        self.input_index_max = input_index_max
        self.learning_rate = learning_rate
        self.player_idx = player_idx
        self.frame_list = frame_list
        self.player_facing_flag = player_facing_flag

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

        # get total size of categorical features
        one_hot_size = 0
        for c, c_list in self.state_format['categorical'].items():
            one_hot_size = len(c_list)
        # sum features, value features + categorical features + last input
        self.model_input_size = (len(self.state_format['values'])*2) + (one_hot_size*2) + (input_index_max+1)
        print(self.model_input_size)

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
            for attrib in self.state_format['values']:  # for each attribute
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
                category_list = [0] * len(self.state_format['categorical'][category])
                category_list[self.state_format['categorical'][category][str(game_state[p_idx][category])]] = 1  # encode
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
        self.model, self.optimizer = model.setup_model(
            actions_size=self.input_index_max + 1,
            input_size=self.model_input_size,
            learning_rate=self.learning_rate
        )

        self.target, _ = model.setup_model(
            actions_size=self.input_index_max + 1,
            input_size=self.model_input_size,
            learning_rate=self.learning_rate
        )

        self.episode_number, self.run_count = eval_util.get_next_episode(player_idx=self.player_idx)
        if not model.load_model(self.model, self.optimizer, self.player_idx):
            print("fresh model")
            torch.manual_seed(0)

            model.save_model(self.model, self.optimizer, self.player_idx)

        self.target.load_state_dict(self.model.state_dict())
        self.model = self.model.to(device)
        self.target = self.target.to(device)

        if not config.settings['last_episode_only']:
            self.reward_paths = eval_util.get_reward_paths(self.player_idx)

        self.target.eval()

        # warm up model
        stress_data = []
        for n in range(0, self.frames_per_evaluation+1):
            stress_data.append([[random.random() for n in range(0, self.model_input_size)]])

        in_tensor = torch.Tensor([stress_data]).to(device)

        # input data into model
        with torch.no_grad():
            for i in range(0, in_tensor.size()[0]):
                out_tensor, hidden_state = self.model(in_tensor[i])

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
            model.save_model(self.model, self.optimizer, self.player_idx)
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

        reward_paths = calc_reward.generate_rewards(
            reward_path=reward_path,
            eval_path=eval_path,
            reward_columns=config.settings['reward_columns'][0],
            falloff=config.settings['reward_falloff'],
            player_idx=self.player_idx,
            reaction_delay=self.reaction_delay,
            atk_preframes=config.settings['atk_preframes'],
            whiff_reward=config.settings['whiff_reward'],
            reward_gamma=config.settings['reward_gamma'],
            frames_per_observation=config.settings['p{}_model'.format(self.player_idx)]['frames_per_observation'],
            stats_path=stats_path,
            episode_number=self.episode_number
        )
        if config.settings['last_episode_only']:
            reward_sample = [reward_paths]
        else:
            self.reward_paths.append(reward_paths)
            if len(self.reward_paths) > config.settings['episode_sample_size']:
                reward_sample = random.sample(self.reward_paths, config.settings['episode_sample_size'])
            else:
                reward_sample = self.reward_paths

        train_rnn_model.train_model(reward_sample, stats_path, self.model, self.target, self.optimizer,
                                    config.settings['epochs'],
                                    self.episode_number, self.frames_per_evaluation)

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
            eps_threshold = initial_epsilon
            esp_count = 0
            no_explore_count = 0
            explore_better_action = config.settings['probability_action']
            self.mean_pred_q = 0
            last_mean_pred_q = self.mean_pred_q
            self.mean_pred_explore_count = 0

            # TODO ACTION SCRIPT
            act_script = action_script.ActionScript()

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
                        if (last_normalized_index - self.reaction_delay) > last_evaluated_index:
                            # create slice for evaluation
                            normalized_states = normalized_states

                            # flatten for input into model
                            flat_frames = []

                            for f_idx in range(last_evaluated_index - self.frames_per_evaluation+1,
                                               last_evaluated_index+1):
                                if f_idx < 0:
                                    flat_frames.append(
                                        [0.0]*len(normalized_states[0]['game']) + [0.0]*len(normalized_inputs[0]))
                                else:
                                    flat_frames.append(
                                        normalized_states[f_idx]['game'] + normalized_inputs[f_idx])
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
                                out_tensor, hidden_state = self.model(in_tensor.unsqueeze(0))

                            detached_out = out_tensor[-1].detach().cpu()

                            if random.random() < eps_threshold:
                                action_index = random.randrange(0, len(detached_out))
                            else:
                                action_index = torch.argmax(detached_out).numpy().item()
                            max_q = detached_out[action_index].numpy().item()


                            # elif explore_better_action:
                            #     self.mean_pred_explore_count = self.mean_pred_explore_count + 1
                            #     out_clone = detached_out.clone()
                            #     if out_clone.min() < 0:
                            #         out_clone = out_clone - out_clone.min()

                                # TODO ACTION
                                # if config.settings['input_mask'] is not None:
                                #     action_index = torch.multinomial(out_clone[config.settings['input_mask']], 1).numpy().item()
                                # else:
                                #     action_index = torch.multinomial(out_clone, 1).numpy().item()

                            # if config.settings['input_mask'] is not None and not explore_better_action:
                            #     max_index = torch.argmax(
                            #         detached_out[config.settings['input_mask']]).numpy().item()
                            #     action_index = config.settings['input_mask'][max_index]
                            #     max_q = detached_out.max().numpy().item()
                            # else:
                            #     action_index = torch.argmax(detached_out).numpy().item()
                            #     max_q = detached_out[action_index].numpy().item()

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

                            self.input_index.value = action_index
                            self.player_facing_flag.value = player_facing_flag

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
                    print("eps={} no_explore={}".format(self.epsilon, no_explore_count))

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
                    last_mean_pred_q = self.mean_pred_q
                    self.mean_pred_q = 0
                    self.mean_pred_explore_count = 0
                    self.run_count = self.run_count + 1
                    logger.debug("{} finished cleanup".format(self.player_idx))
                    act_script.reset()
                    self.eval_status['storing_eval'] = False  # finished storing eval
        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
            raise identifier

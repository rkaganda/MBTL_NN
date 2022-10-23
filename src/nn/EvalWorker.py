import time
import logging
import traceback

import multiprocessing as mp
import numpy as np

import torch
import torch.nn as nn
# import torch.multiprocessing as mp

import mbtl_input
import nn.model as model
import nn.eval_util as eval_util
import config

logging.basicConfig(filename='logs/train.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)


class EvalWorker(mp.Process):
    def __init__(self, game_states, frames_per_evaluation, reaction_delay, die, input_state, state_format,
                 learning_rate, player_idx, frame_list, worker_ready):
        super(EvalWorker, self).__init__()
        self.states = game_states
        self.frames_per_evaluation = frames_per_evaluation
        self.reaction_delay = reaction_delay
        self.die = die
        self.eval_ready = worker_ready
        self.input_state = input_state
        self.learning_rate = learning_rate
        self.player_idx = player_idx
        self.frame_list = frame_list

        self.state_format = state_format

        self.model = None
        self.optimizer = None

    # normalize the state attributes between 0, 1 using precalculated min max
    def normalize_state(self, state):
        norm_state = dict()

        # normalize game state
        minmax = self.state_format['minmax']
        game_state = state['game']
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

        # normalize inputs
        input_state = state['input']
        norm_state['input'] = list()
        for i_ in self.state_format['input']:  # for each input
            norm_state['input'].append(input_state[i_])

        return norm_state

    def setup_model(self):
        self.model, self.optimizer = model.setup_model(
            frames_per_observation=self.frames_per_evaluation,
            input_state_size=len(self.input_state),
            state_state_size=len(self.state_format['attrib']),
            learning_rate=self.learning_rate
        )

        if config.settings['model_file'] is not None:
            model.load_model(self.model, self.optimizer)
            print("loaded model")
        else:
            print("fresh model")

        self.model.to(device)
        self.model.eval()

        # warm up model
        in_tensor = torch.ones(
            self.frames_per_evaluation * (len(self.input_state) + (len(self.state_format['attrib']) * 2))).to(device)
        out_tensor = self.model(in_tensor)

    def round_cleanup(self, normalized_states, model_output):
        eval_util.store_eval_output(
            normalized_states,
            self.states,
            model_output,
            self.state_format
        )

    def run(self):
        self.setup_model()

        try:
            normalized_states = list()
            model_output = dict()
            last_normalized_index = 0
            last_evaluated_index = 0
            self.eval_ready.set()

            while not self.die.is_set():
                if len(self.states) > len(normalized_states):  # if there are game states to normalize
                    # normalize a frame and append to to normalized states
                    normalized_states.append(
                        self.normalize_state(self.states[last_normalized_index])
                    )
                    last_normalized_index = last_normalized_index + 1

                    # if reaction time has passed, and we have enough frames to eval
                    if ((last_normalized_index - self.reaction_delay) >= last_evaluated_index) and (
                            (len(normalized_states) - self.reaction_delay) >= self.frames_per_evaluation):

                        # create slice for evaluation
                        evaluation_frames = normalized_states[:-self.reaction_delay]
                        evaluation_frames = evaluation_frames[-self.frames_per_evaluation:]

                        # flatten for input into model
                        flat_frames = []
                        for f_ in evaluation_frames:
                            flat_frames = flat_frames + f_['input'] + f_['game']

                        # create tensor
                        in_tensor = torch.Tensor(flat_frames).to(device)

                        # input data into model
                        with torch.no_grad():
                            out_tensor = self.model(in_tensor)

                        detached_out = out_tensor.detach().cpu()
                        prob = torch.bernoulli(detached_out).numpy()

                        for idx, key in enumerate(self.state_format['input']):
                            self.input_state[key] = 1 if prob[idx] == 1 else 0

                        # store model output
                        model_output[last_evaluated_index] = {
                            'output': list(detached_out.numpy()),
                            'frame': self.frame_list[-1],
                            'norm_frame': last_normalized_index,
                            'window': [
                                last_normalized_index - self.reaction_delay - self.frames_per_evaluation,
                                last_normalized_index - self.reaction_delay - 1
                            ],
                            # 'input_tensor': flat_frames
                        }

                        # increment evaluated index
                        last_evaluated_index = last_evaluated_index + 1
            print("cleaning up")
            self.round_cleanup(normalized_states, model_output)
            if config.settings['save_model']:
                model.save_model(self.model, self.optimizer)
        except Exception as identifier:
            logger.error(identifier)
            logger.error(traceback.format_exc())
            raise identifier

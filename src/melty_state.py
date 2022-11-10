import copy

import config
import json

# load minmax
with open("{}/{}".format(config.settings['data_path'], config.settings['minmax_file'])) as f:
    state_format = json.load(f)


def get_minmax():
    minmax = state_format['minmax']
    minmax['x_spac'] = dict()
    minmax['x_spac']['max'] = abs(state_format['minmax']['x_posi']['max'] - state_format['minmax']['x_posi']['min'])
    minmax['x_spac']['min'] = 0
    minmax['y_spac'] = dict()
    minmax['y_spac']['max'] = abs(state_format['minmax']['y_posi']['max'] - state_format['minmax']['y_posi']['min'])
    minmax['y_spac']['min'] = 0

    return minmax


def get_attributes():
    state_format['game_attrib'] = copy.deepcopy(state_format['attrib'])
    state_format['attrib'].append('x_spac')
    state_format['attrib'].append('y_spac')

    return state_format['attrib'], state_format['game_attrib']


def calc_extra_states(game_state):
    for p_idx in [0, 1]:
        game_state[p_idx]['x_spac'] = abs(game_state[p_idx]['x_posi'] - game_state[1-p_idx]['x_posi'])
        game_state[p_idx]['y_spac'] = abs(game_state[p_idx]['y_posi'] - game_state[1-p_idx]['y_posi'])

    return game_state

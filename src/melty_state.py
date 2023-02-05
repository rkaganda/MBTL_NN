import copy

import config
import json

# load minmax
with open("{}/{}".format(config.settings['data_path'], config.settings['minmax_file'])) as f:
    state_format = json.load(f)


def get_minmax():
    minmax = state_format['minmax']

    # create state the stores opponents relative position
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


def encode_relative_states(game_state, player_idx):
    player_facing_flag = 1

    if game_state[player_idx]['x_posi'] < game_state[1 - player_idx]['x_posi']:
        player_facing_flag = 0

    for p_idx in [0, 1]:
        # encode opponents position relative to player
        game_state[p_idx]['x_spac'] = abs(game_state[p_idx]['x_posi'] - game_state[1 - p_idx]['x_posi'])
        game_state[p_idx]['y_spac'] = abs(game_state[p_idx]['y_posi'] - game_state[1 - p_idx]['y_posi'])

    if player_facing_flag == 1:
        x_distance_from_right = state_format['minmax']['x_posi']['max'] - game_state[player_idx]['x_posi']
        game_state[player_idx]['x_posi'] = state_format['minmax']['x_posi']['min'] + x_distance_from_right

        x_distance_from_left = state_format['minmax']['x_posi']['min'] - game_state[1 - player_idx]['x_posi']
        game_state[1 - player_idx]['x_posi'] = state_format['minmax']['x_posi']['max'] + x_distance_from_left

    if player_idx != 0:
        game_state = copy.deepcopy([game_state[1], game_state[0]])
    else:
        game_state = copy.deepcopy([game_state[0], game_state[1]])

    return game_state, player_facing_flag

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

    # add minmax that doesn't exist
    minmax['player_facing_flag'] = dict()
    minmax['player_facing_flag']['min'] = 0
    minmax['player_facing_flag']['max'] = 1

    return minmax


def get_attributes():
    state_format['game_attrib'] = copy.deepcopy(state_format['attrib'])
    state_format['attrib'].append('x_spac')
    state_format['attrib'].append('y_spac')
    state_format['attrib'].append('player_facing_flag')

    return state_format['attrib'], state_format['game_attrib']


def encode_relative_states(game_state, player_idx):
    for p_idx in [0, 1]:
        # encode opponents position relative to player
        game_state[p_idx]['x_spac'] = abs(game_state[p_idx]['x_posi'] - game_state[1 - p_idx]['x_posi'])
        game_state[p_idx]['y_spac'] = abs(game_state[p_idx]['y_posi'] - game_state[1 - p_idx]['y_posi'])

    for p_idx in [0, 1]:
        if game_state[p_idx]['x_posi'] < game_state[1-p_idx]['x_posi']:
            game_state[p_idx]['player_facing_flag'] = 0
        else:
            game_state[p_idx]['player_facing_flag'] = 1

    for p_idx in [0, 1]:
        # encode x position relative to left side
        if game_state[p_idx]['x_posi'] > 0:  # if player is right of center (0)
            # swap player to left
            game_state[p_idx]['x_posi'] = game_state[p_idx]['x_posi'] * -1
            # move opponent relative to player on left side
            if game_state[p_idx]['player_facing_flag']:
                game_state[1-p_idx]['x_posi'] = game_state[1-p_idx]['x_posi'] + game_state[p_idx]['x_spac']
            else:
                game_state[1-p_idx]['x_posi'] = game_state[1-p_idx]['x_posi'] - game_state[p_idx]['x_spac']

    return game_state

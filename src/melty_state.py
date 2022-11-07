import config
import json

# load minmax
with open(config.settings['minmax_file']) as f:
    state_format = json.load(f)


def get_minmax():
    minmax = state_format['minmax']
    minmax['x_spacing'] = state_format['minmax']['x_position']['max'] - state_format['minmax']['x_position']['min']
    minmax['y_spacing'] = state_format['minmax']['y_position']['max'] - state_format['minmax']['y_position']['min']

    return minmax


def get_attributes():
    attrib = state_format['attrib']
    attrib.append('x_spacing')
    attrib.append('y_spacing')
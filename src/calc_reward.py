import pandas as pd
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
import datetime
import logging
import itertools
import numpy as np

import config

logging.basicConfig(filename='./logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def load_file(file_path):
    with open(file_path) as f:
        file_dict = json.load(f)

    return file_dict


def create_eval_df(file_dict):
    # reform model_output
    d_ = dict()

    for frame_idx, (frame_key, frame_row) in enumerate(file_dict['model_output'].items()):
        row = dict()
        for row_idx, (key, item) in enumerate(frame_row.items()):
            if isinstance(item, list):
                for idx, value in enumerate(item):
                    row["{}_{}".format(key, idx)] = value
            else:
                row["{}".format(key)] = item
        d_[frame_idx] = row

    eval_df = pd.DataFrame.from_dict(d_, orient='index')

    return eval_df


def create_norm_state_df(file_dict):
    norm_dict = {}

    for index, item in enumerate(file_dict['normalized_states']):
        row = dict()
        for idx, value in enumerate(item['input'] + item['game']):
            row[idx] = value
        norm_dict[index] = row

    norm_state_df = pd.DataFrame.from_dict(norm_dict, orient='index')

    return norm_state_df


def calcuate_actual_state_df(file_dict):
    actual_state_dict = {}

    for index, item in enumerate(file_dict['states']):
        row = dict()
        for player_id, player_states in item['game'].items():
            for attrib, value in player_states.items():
                row["p_{}_{}".format(player_id, attrib)] = value

        row["input"] = item['input']
        actual_state_dict[index] = row

    actual_state_df = pd.DataFrame.from_dict(actual_state_dict, orient='index')

    return actual_state_df


def calculate_reformed_input_df(eval_df, norm_state_df):
    input_windows = list()
    for idx, row in eval_df.iterrows():
        input_window = list()
        for idx in range(int(row['window_0']), int(row['window_1']) + 1):
            input_window = input_window + norm_state_df.iloc[idx].to_list()
        input_windows.append(input_window)

    reformed_input_df = pd.DataFrame(input_windows)
    reformed_input_df.index = list(reformed_input_df.index)
    reformed_input_df.columns = list(reformed_input_df.columns)
    reformed_input_df.columns = reformed_input_df.columns.map(str)
    reformed_input_df = reformed_input_df.add_prefix("state_")

    return reformed_input_df


def create_eval_state_df(eval_df, actual_state_df, reformed_input_df):
    eval_state_df = eval_df.merge(
        actual_state_df,
        how='left',
        left_on='window_0',
        right_index=True
    )

    eval_state_df = eval_state_df.merge(
        reformed_input_df,
        right_index=True,
        left_index=True
    )

    return eval_state_df


def generate_diff(eval_state_df, reward_columns):
    for c, modifer in reward_columns.items():
        eval_state_df['{}_diff'.format(c)] = eval_state_df[c].diff()
        eval_state_df['{}_diff'.format(c)] = eval_state_df['{}_diff'.format(c)] * modifer

    return eval_state_df


def trim_reward_df(eval_state_df, reward_column):
    eval_state_df.rename(columns={reward_column: "reward"}, inplace=True)
    state_columns = [c for c in eval_state_df.columns if c.startswith('state')]
    eval_state_df = eval_state_df[state_columns + ['reward'] + ['input']]

    eval_state_df = eval_state_df[:-1]

    return eval_state_df


def calculate_reward_from_eval(
        file_path, reward_columns, falloff, player_idx, reaction_delay, hit_preframes, reward_gamma):
    file_dict = load_file(file_path)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    eval_df = create_eval_df(file_dict)

    norm_state_df = create_norm_state_df(file_dict)

    actual_state_df = calcuate_actual_state_df(file_dict)

    reformed_input_df = calculate_reformed_input_df(eval_df, norm_state_df)

    eval_state_df = create_eval_state_df(eval_df, actual_state_df, reformed_input_df)

    eval_state_df = generate_diff(eval_state_df, reward_columns)

    eval_state_df['reward'] = 0

    eval_state_df = apply_hit_segment_rewards(player_idx, eval_state_df, hit_preframes)

    eval_state_df = apply_reward_discount(eval_state_df, reward_gamma)

    output_with_input_and_reward = trim_reward_df(eval_state_df, 'reward_total_norm')

    return output_with_input_and_reward


def apply_reward_discount(df, gamma):
    df['discounted_reward'] = df['reward']
    for n in range(0, 5):
        df['discounted_reward'] = (df['discounted_reward'] + df['discounted_reward'].shift(-1) * gamma) * .5

        df['actual_reward'] = df.apply(
            lambda x: x['reward'] if abs(x['reward']) > abs(x['discounted_reward']) else x['discounted_reward'], axis=1)

    return df


def apply_hit_segment_rewards(p_idx, df, hit_preframes):
    # get start and stop of of hit
    for p_i in range(0, 2):
        hit_col = 'p_{}_hit'.format(p_i)
        diff_col = 'p_{}_health_diff'.format(p_i)

        # calculate hit changes
        v = (df[hit_col] != df[hit_col].shift()).cumsum()
        u = df.groupby(v)[hit_col].agg(['all', 'count'])
        m = u['all'] & u['count'].ge(1)

        # create hit segments
        hit_segements = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]

        health_lost_segs = {}
        for hs in hit_segements:
            # calcluate reward for entire hit segment
            health_lost_segs[hs] = df[(df.index >= hs[0]) & (df.index <= hs[1])][diff_col].sum()

        for idxs, val in health_lost_segs.items():  # for each hit segment
            # apply reward to frames from hit start - hit_preframes to hit end
            df.loc[(df.index <= idxs[1]) & (df.index > idxs[0] - hit_preframes), 'reward'] = val

        # if p_i == 1 - p_idx:
        #     for hs in hit_segements:
        #         df.loc[(df.index<(hs-20))&(df.index>idx-20),'reward'] = df.loc[(df.index<idx)&(df.index>idx-20),'reward'] - 1000
    return df


def apply_invalid_input_reward(e_df, player_idx, reaction_delay, neutral_index):
    invalid_frame_window = 5
    motion_change_col = 'p_{}_motion_change'.format(player_idx)
    motion_type_col = 'p_{}_motion_type'.format(player_idx)

    e_df[motion_change_col] = e_df[motion_type_col].diff()
    # change_index = e_df[e_df[motion_change_col] != 0].index.tolist()
    change_index = e_df[e_df[motion_change_col] != 0].index.tolist()

    valid_input_index = [v - 1 for v in change_index]
    neutral_inputs = e_df[e_df['input'] == neutral_index].index.tolist()

    valid_windows = set(
        itertools.chain.from_iterable([list(range(v - invalid_frame_window, v)) for v in valid_input_index])).union(
        neutral_inputs)
    valid_windows = [v for v in valid_windows if v > 0]
    invalid_index = e_df[~e_df.index.isin(valid_windows)].index.tolist()
    invalid_index = [v for v in invalid_index if v > reaction_delay]

    e_df.loc[e_df.index[invalid_index], 'reward_total_norm'] = -1000

    return e_df


def generate_json_from_in_out_df(output_with_input_and_reward):
    json_dict = {}

    state_columns = [c for c in output_with_input_and_reward.columns if c.startswith('state_')]
    action_columns = [c for c in output_with_input_and_reward.columns if c.startswith('input')]
    json_dict['state'] = output_with_input_and_reward[state_columns].values.tolist()
    json_dict['reward'] = output_with_input_and_reward['reward'].values.tolist()
    json_dict['action'] = output_with_input_and_reward[action_columns].values.tolist()

    return json_dict


def generate_rewards(eval_path, reward_path, reward_columns, falloff, player_idx, reaction_delay, hit_preframes, reward_gamma):
    print(reward_path)
    Path("{}".format(reward_path)).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]

    for file in onlyfiles:
        try:
            if file.startswith('eval_'):
                reward_file = file.replace('eval_', 'reward_')
                file_name = "{}/{}".format(eval_path, file)
                df = calculate_reward_from_eval(file_name, reward_columns, falloff, player_idx, reaction_delay,
                                                hit_preframes, reward_gamma)
                file_json = generate_json_from_in_out_df(df)

                with open("{}/{}".format(reward_path, reward_file), 'w') as f_writer:
                    f_writer.write(json.dumps(file_json))

        except Exception as e:
            logger.debug("file_name={}/{}".format(eval_path, file))

    return [reward_path]


def main():
    pass

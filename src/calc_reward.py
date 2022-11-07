import pandas as pd
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
import datetime
import logging

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
        for idx, value in enumerate([item['input']] + item['game']):
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


def apply_reward(eval_state_df, reward_column, reward_falloff):
    reward_values = {}
    eval_state_df["{}_reward".format(reward_column)] = 0
    eval_state_df[reward_column].fillna(0, inplace=True)

    for idx, row in eval_state_df[::-1].iterrows():
        reward_values[idx] = {
            "fall_off": (row[reward_column] / reward_falloff),
            "interval": 0,
            "value": row[reward_column]
        }
        actual_reward = 0
        remove_idx = []
        for r_key, r_item in reward_values.items():
            actual_reward = actual_reward + r_item['value'] - (r_item['fall_off'] * r_item['interval'])

            if r_item['interval'] == reward_falloff:
                remove_idx.append(r_key)
            else:
                r_item['interval'] = r_item['interval'] + 1
        eval_state_df.at[idx, "{}_reward".format(reward_column)] = actual_reward

        for rem_idx in remove_idx:
            del reward_values[rem_idx]


def calculate_reward_df(eval_state_df, reward_columns, falloff):
    reward_c = []

    for idx, col in enumerate(reward_columns):
        rc = "{}_diff".format(col)
        apply_reward(eval_state_df, reward_column=rc, reward_falloff=falloff)
        reward_c.append("{}_reward".format(rc))

    eval_state_df['reward_total'] = 0

    for r in reward_c:
        eval_state_df['reward_total'] = eval_state_df['reward_total'] + \
                                                      eval_state_df[r]

    # eval_state_df['reward_total_norm'] = \
    #     (eval_state_df['reward_total'] - eval_state_df['reward_total'].mean()) / \
    #     eval_state_df['reward_total'].std()

    eval_state_df['reward_total_norm'] = eval_state_df['reward_total']

    # eval_state_df['reward_total_norm'] = \
    #     (eval_state_df['reward_total'] - eval_state_df['reward_total'].min())/ \
    #     (eval_state_df['reward_total'].max() - eval_state_df['reward_total'].min())

    eval_state_df['reward_total_norm'].fillna(0, inplace=True)

    return eval_state_df


def trim_reward_df(eval_state_df, reward_column):
    eval_state_df.rename(columns={reward_column: "reward"}, inplace=True)
    state_columns = [c for c in eval_state_df.columns if c.startswith('state')]
    eval_state_df = eval_state_df[state_columns + ['reward'] + ['input']]

    eval_state_df = eval_state_df[:-1]

    return eval_state_df


def caclulate_reward_from_eval(file_path, reward_columns, falloff):
    file_dict = load_file(file_path)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    eval_df = create_eval_df(file_dict)

    norm_state_df = create_norm_state_df(file_dict)

    actual_state_df = calcuate_actual_state_df(file_dict)

    reformed_input_df = calculate_reformed_input_df(eval_df, norm_state_df)

    eval_state_df = create_eval_state_df(eval_df, actual_state_df, reformed_input_df)

    eval_state_df = generate_diff(eval_state_df, reward_columns)

    eval_state_df = calculate_reward_df(eval_state_df, list(reward_columns.keys()), falloff)

    output_with_input_and_reward = trim_reward_df(eval_state_df, 'reward_total_norm')

    return output_with_input_and_reward


def generate_json_from_in_out_df(output_with_input_and_reward):
    json_dict = {}

    state_columns = [c for c in output_with_input_and_reward.columns if c.startswith('state_')]
    action_columns = [c for c in output_with_input_and_reward.columns if c.startswith('input')]
    json_dict['state'] = output_with_input_and_reward[state_columns].values.tolist()
    json_dict['reward'] = output_with_input_and_reward['reward'].values.tolist()
    json_dict['action'] = output_with_input_and_reward[action_columns].values.tolist()

    return json_dict


def generate_rewards(eval_path, reward_path, reward_columns, falloff):
    print(reward_path)
    Path("{}".format(reward_path)).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]

    for file in onlyfiles:
        try:
            if file.startswith('eval_'):
                reward_file = file.replace('eval_', 'reward_')
                file_name = "{}/{}".format(eval_path, file)
                df = caclulate_reward_from_eval(file_name, reward_columns, falloff)
                file_json = generate_json_from_in_out_df(df)

                with open("{}/{}".format(reward_path, reward_file), 'w') as f_writer:
                    f_writer.write(json.dumps(file_json))
        except Exception as e:
            logger.debug("file_name={}/{}".format(eval_path, file))


def main():
    pass
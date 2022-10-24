import pandas as pd
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path


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
        for p_input, value in item['input'].items():
            row["input_{}".format(p_input)] = value
        actual_state_dict[index] = row

    actual_state_df = pd.DataFrame.from_dict(actual_state_dict, orient='index')

    return actual_state_df


def calculate_reformed_input_df(eval_df, norm_state_df):
    input_windows = list()
    for jdx, row in eval_df.iterrows():
        input_window = list()
        for idx in range(int(row['window_0']), int(row['window_1']) + 1):
            input_window = input_window + norm_state_df.iloc[idx].to_list()
        input_windows.append(input_window)

    reformed_input_df = pd.DataFrame(input_windows)
    reformed_input_df.index = list(reformed_input_df.index)
    reformed_input_df.columns = list(reformed_input_df.columns)
    reformed_input_df.columns = reformed_input_df.columns.map(str)

    return reformed_input_df


def create_merged_eval_actual_state_df(eval_df, actual_state_df):
    merged_eval_actual_state_df = eval_df.merge(
        actual_state_df,
        how='left',
        left_on='window_0',
        right_index=True
    )

    return merged_eval_actual_state_df


def generate_diff(merged_eval_actual_state_df, reward_columns):
    for c, modifer in reward_columns.items():
        merged_eval_actual_state_df['{}_diff'.format(c)] = merged_eval_actual_state_df[c].diff()
        merged_eval_actual_state_df['{}_diff'.format(c)] = merged_eval_actual_state_df['{}_diff'.format(c)] * modifer

    return merged_eval_actual_state_df


def apply_reward(merged_eval_actual_state_df, reward_column, reward_falloff):
    reward_values = {}
    merged_eval_actual_state_df["{}_reward".format(reward_column)] = 0
    merged_eval_actual_state_df[reward_column].fillna(0, inplace=True)

    for idx, row in merged_eval_actual_state_df[::-1].iterrows():
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
        merged_eval_actual_state_df.at[idx, "{}_reward".format(reward_column)] = actual_reward

        for rem_idx in remove_idx:
            del reward_values[rem_idx]


def calculate_reward_df(merged_eval_actual_state_df, reward_columns, falloff):
    reward_c = []

    for idx, col in enumerate(reward_columns):
        rc = "{}_diff".format(col)
        apply_reward(merged_eval_actual_state_df, reward_column=rc, reward_falloff=falloff)
        reward_c.append("{}_reward".format(rc))

    merged_eval_actual_state_df['reward_total'] = 0

    for r in reward_c:
        merged_eval_actual_state_df['reward_total'] = merged_eval_actual_state_df['reward_total'] + \
                                                      merged_eval_actual_state_df[r]

    merged_eval_actual_state_df['reward_total_norm'] = \
        (merged_eval_actual_state_df['reward_total'] - merged_eval_actual_state_df['reward_total'].mean()) / \
        merged_eval_actual_state_df['reward_total'].std()

    return merged_eval_actual_state_df


def merge_input_with_reward(merged_eval_actual_state_df, reformed_input_df, reward_column):
    merged_eval_actual_state_df.rename(columns={reward_column: "reward"}, inplace=True)
    reformed_input_df = reformed_input_df.add_prefix('input_')
    output_columns = [c for c in merged_eval_actual_state_df.columns if c.startswith('output')]
    merged_eval_actual_state_df = merged_eval_actual_state_df[output_columns + ['reward']]

    output_with_input_and_reward = merged_eval_actual_state_df.merge(
        reformed_input_df,
        right_index=True,
        left_index=True
    )

    output_with_input_and_reward = output_with_input_and_reward[:-1]

    return output_with_input_and_reward


def caclulate_reward_from_eval(file_path, reward_columns, falloff):
    print("reward_columns={}".format(reward_columns))
    print("falloff={}".format(falloff))
    file_dict = load_file(file_path)

    eval_df = create_eval_df(file_dict)

    norm_state_df = create_norm_state_df(file_dict)

    actual_state_df = calcuate_actual_state_df(file_dict)

    reformed_input_df = calculate_reformed_input_df(eval_df, norm_state_df)

    merged_eval_actual_state_df = create_merged_eval_actual_state_df(eval_df, actual_state_df)

    merged_eval_actual_state_df = generate_diff(merged_eval_actual_state_df, reward_columns)

    merged_eval_actual_state_df = calculate_reward_df(merged_eval_actual_state_df, list(reward_columns.keys()), falloff)

    output_with_input_and_reward = merge_input_with_reward(merged_eval_actual_state_df, reformed_input_df,
                                                           'reward_total_norm')

    return output_with_input_and_reward


def generate_json_from_in_out_df(output_with_input_and_reward):
    json_dict = {}

    output_columns = [c for c in output_with_input_and_reward.columns if c.startswith('output_')]
    input_columns = [c for c in output_with_input_and_reward.columns if c.startswith('input_')]
    json_dict['output'] = output_with_input_and_reward[output_columns].values.tolist()
    json_dict['reward'] = output_with_input_and_reward['reward'].values.tolist()
    json_dict['input'] = output_with_input_and_reward[input_columns].values.tolist()

    return json_dict


def main():
    run = "p1_vs_cpu_heath"

    eval_path = "data/eval/{}/evals".format(run)
    reward_path = "data/eval/{}/reward".format(run)

    reward_columns = {
        "p_0_health": -1,
        "p_1_health": 1
    }
    falloff = 15

    Path("{}".format(reward_path)).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]

    for file in onlyfiles:
        if file.startswith('eval_'):
            reward_file = file.replace('eval_', 'reward_')
            file_name = "{}/{}".format(eval_path, file)
            df = caclulate_reward_from_eval(file_name, reward_columns, falloff)
            file_json = generate_json_from_in_out_df(df)

            with open("{}/{}".format(reward_path, reward_file), 'w') as f_writer:
                f_writer.write(json.dumps(file_json))


if __name__ == "__main__":
    main()




import pandas as pd
import json
from IPython.display import JSON
from os import listdir
from os.path import isfile, join

import plotly.express as px
import plotly.graph_objs as go
from plotly.graph_objs.scatter.marker import Line

pd.options.display.max_rows = 4000
pd.options.display.max_seq_items = 4000
pd.options.display.max_columns = 4000


def load_file(file_path):
    with open(file_path) as f:
        file_dict = json.load(f)

    JSON(file_dict)

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

    # JSON(d_)

    eval_df = pd.DataFrame.from_dict(d_, orient='index')

    # display(eval_df.head())
    # display(eval_df.tail())

    return eval_df


def create_norm_state_df(file_dict):
    norm_dict = {}

    for index, item in enumerate(file_dict['normalized_states']):
        row = dict()
        for idx, value in enumerate(item['input'] + item['game']):
            row[idx] = value
        norm_dict[index] = row

    # JSON(norm_dict)

    norm_state_df = pd.DataFrame.from_dict(norm_dict, orient='index')

    # display(norm_state_df.head())
    # display(norm_state_df.tail())

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

    # JSON(actual_state_dict)

    actual_state_df = pd.DataFrame.from_dict(actual_state_dict, orient='index')

    ##display(actual_state_df.head())
    ##display(actual_state_df.tail())

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

    ##display(reformed_input_df.head())
    ##display(reformed_input_df.tail())

    return reformed_input_df


def create_merged_eval_actual_state_df(eval_df, actual_state_df):
    merged_eval_actual_state_df = eval_df.merge(
        actual_state_df,
        how='left',
        left_on='window_0',
        right_index=True
    )

    ##display(merged_eval_actual_state_df.head())
    # display(merged_eval_actual_state_df[merged_eval_actual_state_df.isna().any(axis=1)])

    return merged_eval_actual_state_df


def generate_diff(merged_eval_actual_state_df):
    # merged_eval_actual_state_df['p_0_x_posi_diff'] = file_dict['state_format']['minmax']['x_posi']['max'] - merged_eval_actual_state_df['p_0_x_posi']
    merged_eval_actual_state_df['p_0_x_posi_diff'] = merged_eval_actual_state_df['p_0_x_posi'].diff()
    merged_eval_actual_state_df['p_0_x_posi_diff'] = merged_eval_actual_state_df['p_0_x_posi_diff'].shift(-1)
    # display(merged_eval_actual_state_df[merged_eval_actual_state_df.isna().any(axis=1)])

    return merged_eval_actual_state_df


def plot_diff(merged_eval_actual_state_df):
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=merged_eval_actual_state_df.index,
        y=merged_eval_actual_state_df['p_0_x_posi'],
        mode='lines',
        name='p_0_x_posi'))

    # fig.show()

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Bar(
        x=merged_eval_actual_state_df.index,
        y=merged_eval_actual_state_df['p_0_x_posi_diff'],
        # mode='lines',
        name='p_0_x_posi_diff'))

    # fig.show()


def apply_reward(merged_eval_actual_state_df, reward_column, reward_falloff):
    reward_values = {}
    merged_eval_actual_state_df["{}_reward".format(reward_column)] = 0
    merged_eval_actual_state_df[reward_column].fillna(0,inplace=True)

    for idx, row in merged_eval_actual_state_df[::-1].iterrows():
        reward_values[idx] = {
            "fall_off": (row[reward_column]/reward_falloff),
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

    return merged_eval_actual_state_df


def calculate_reward_df(merged_eval_actual_state_df):
    reward_columns = ['p_0_x_posi_diff']
    falloff = 5

    for col in reward_columns:
        merged_eval_actual_state_df = apply_reward(merged_eval_actual_state_df, reward_column=col, reward_falloff=falloff)

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Bar(
        x=merged_eval_actual_state_df.index,
        y=merged_eval_actual_state_df['p_0_x_posi_diff'],
        # mode='lines',
        name='p_0_x_posi_diff'))

    # Add traces
    fig.add_trace(go.Scatter(
        x=merged_eval_actual_state_df.index,
        y=merged_eval_actual_state_df['p_0_x_posi_diff_reward'],
        mode='lines',
        name='p_0_x_posi_diff_reward'))

    # fig.show()
    merged_eval_actual_state_df['p_0_x_posi_diff_reward'] = \
        (merged_eval_actual_state_df['p_0_x_posi_diff_reward'] - merged_eval_actual_state_df[
            'p_0_x_posi_diff_reward'].mean()) / merged_eval_actual_state_df['p_0_x_posi_diff_reward'].std()

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


def caclulate_reward_from_eval(file_path):
    file_dict = load_file(file_path)

    eval_df = create_eval_df(file_dict)

    norm_state_df = create_norm_state_df(file_dict)

    actual_state_df = calcuate_actual_state_df(file_dict)

    reformed_input_df = calculate_reformed_input_df(eval_df, norm_state_df)

    merged_eval_actual_state_df = create_merged_eval_actual_state_df(eval_df, actual_state_df)

    merged_eval_actual_state_df = generate_diff(merged_eval_actual_state_df)

    plot_diff(merged_eval_actual_state_df)

    merged_eval_actual_state_df = calculate_reward_df(merged_eval_actual_state_df)

    output_with_input_and_reward = merge_input_with_reward(merged_eval_actual_state_df, reformed_input_df,
                                                           'p_0_x_posi_diff_reward')

    #     display(merged_eval_actual_state_df.head())
    #     display(reformed_input_df.head())
    #     display(output_with_input_and_reward.head())
    #     display(output_with_input_and_reward.tail())

    #     display(output_with_input_and_reward.head())
    #     display(output_with_input_and_reward.tail())

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
    run_name = "p1_x_posi_test"
    eval_path = "data/eval/{}".format(run_name)

    onlyfiles = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]

    for file in onlyfiles:
        if file.startswith('eval_'):
            print(file)
            reward_file = file.replace('eval_', 'reward_')
            df = caclulate_reward_from_eval("{}/{}".format(eval_path, file))
            file_json = generate_json_from_in_out_df(df)

            with open("{}/{}".format(eval_path, reward_file), 'w') as f_writer:
                f_writer.write(json.dumps(file_json))


if __name__ == "__main__":
    main()
import pandas as pd
import json
from os import listdir
from os.path import isfile, join
from pathlib import Path
import logging
import itertools

logging.basicConfig(filename='../logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def load_file(file_path: str) -> dict:
    """
    # load json file from path

    :param file_path:
    :return:
    """
    with open(file_path) as f:
        file_dict = json.load(f)

    return file_dict


def create_eval_df(file_dict: dict) -> pd.DataFrame:
    """
    creates eval dataframe that contains the processing information for each frame
    :param file_dict:
    :return:
    """
    # reform model_output
    d_ = dict()

    for frame_idx, (frame_key, frame_row) in enumerate(file_dict['model_output'].items()):
        row = dict()
        for row_idx, (key, item) in enumerate(frame_row.items()):
            if isinstance(item, list):
                for idx, value in enumerate(item):  # create a column for each list item
                    row["{}_{}".format(key, idx)] = value
            else:
                row["{}".format(key)] = item
        d_[frame_idx] = row

    eval_df = pd.DataFrame.from_dict(d_, orient='index')

    return eval_df


def create_norm_state_df(file_dict: dict) -> pd.DataFrame:
    """
    creates dataframe that contains the normalized action (called input) and game state for each frame
    :param file_dict:
    :return:
    """
    norm_dict = {}

    for index, item in enumerate(file_dict['normalized_states']):
        row = dict()
        for idx, value in enumerate(item['input'] + item['game']):
            row[idx] = value
        norm_dict[index] = row

    norm_state_df = pd.DataFrame.from_dict(norm_dict, orient='index')

    return norm_state_df


def calculate_actual_state_df(file_dict: dict) -> pd.DataFrame:
    """
    # creates a dataframe the contains the actual (not normalized) state and action (input) for each frame
    :param file_dict:
    :return:
    """
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


def calculate_reformed_input_df(eval_df: pd.DataFrame, norm_state_df: pd.DataFrame) -> pd.DataFrame:
    """
    merge the eval and normalized state dataframes such that each row represent an evaluation frame

    :param eval_df:
    :param norm_state_df:
    :return:
    """
    input_windows = list()
    for idx, row in eval_df.iterrows():
        input_window = list()
        # the model evaluates multiple frames of data each frame
        # window_0 and window_1 are index of the start and end frames of data for that evaluation frame
        for idx in range(int(row['window_0']), int(row['window_1']) + 1):
            input_window = input_window + norm_state_df.iloc[idx].to_list()
        input_windows.append(input_window)

    reformed_input_df = pd.DataFrame(input_windows)
    reformed_input_df.index = list(reformed_input_df.index)
    reformed_input_df.columns = list(reformed_input_df.columns)
    reformed_input_df.columns = reformed_input_df.columns.map(str)
    reformed_input_df = reformed_input_df.add_prefix("state_")

    return reformed_input_df


def create_eval_state_df(eval_df: pd.DataFrame, actual_state_df: pd.DataFrame) -> pd.DataFrame:
    """
    merge all the dataframes together so that a single row contains
    the multiple frames of normalized state evaluated that frame
    the actual game state on that frame
    the output/action of the model for that frame
    :param eval_df:
    :param actual_state_df:
    :return:
    """
    eval_state_df = eval_df.merge(
        actual_state_df,
        how='left',
        left_on='window_0',
        right_index=True
    )

    return eval_state_df


def generate_diff(eval_state_df: pd.DataFrame, reward_columns: dict) -> pd.DataFrame:
    """
    create diff columns for the change between frames/rows that will be used to calculate reward

    :param eval_state_df:
    :param reward_columns:
    :return:
    """
    for c, modifer in reward_columns.items():
        eval_state_df['{}_diff'.format(c)] = eval_state_df[c].diff()
        eval_state_df['{}_diff'.format(c)] = eval_state_df['{}_diff'.format(c)] * modifer

    return eval_state_df


def trim_reward_df(df: pd.DataFrame, reward_column: str, reaction_delay: int) -> pd.DataFrame:
    """
    remove all columns not needed to generate reward file
    :param df:
    :param reward_column:
    :param reaction_delay:
    :return:
    """
    state_columns = [c for c in df.columns if c.startswith('state')]
    df = df[state_columns + [reward_column] + ['input']]
    df = df.rename(columns={reward_column: "reward"})

    df = df[:-1]

    # z-score
    df['reward'] = df['reward'] / 4000
    df['reward'] = df['reward'].shift(-reaction_delay)

    df = df.dropna()

    return df


def calculate_reward_from_eval(
        file_path: str,
        reward_columns: dict,
        falloff: int,
        player_idx: int,
        reaction_delay: int,
        atk_preframes: int,
        whiff_reward: float,
        reward_gamma: float
) -> pd.DataFrame:
    """
    Generate a json reward file from an eval json file

    :param file_path: the file path of the json eval file
    :param reward_columns: dict containing the actual state columns used to generate reward
    :param falloff: old gamma, not used
    :param player_idx: the idx of the player that the reward is being generated for [0,1]
    :param reaction_delay: the reaction delay, not used
    :param atk_preframes: how many input frames before atk starts to apply whiff reward
    :param whiff_reward: negative whiff reward
    :param reward_gamma: reward gama
    :return: dataframe containing just state, action and reward
    """
    file_dict = load_file(file_path)  # load the eval file
    eval_df = create_eval_df(file_dict)  # contains the eval each frame as a row
    norm_state_df = create_norm_state_df(file_dict)  # contains normalize state for each frame
    actual_state_df = calculate_actual_state_df(file_dict)  # contains actual state for each frame

    eval_state_df = create_eval_state_df(  # actual state, norm state, eval info
        eval_df, actual_state_df)

    eval_state_df = generate_diff(eval_state_df, reward_columns)  # generate diff columns to calc reward

    eval_state_df['reward'] = 0  # set reward each frame to 0

    # apply rewards for hitting and whiffing
    eval_state_df = apply_motion_type_reward(eval_state_df, atk_preframes, whiff_reward)

    # apply rewards for getting hit
    eval_state_df = apply_negative_motion_type_reward(eval_state_df, atk_preframes, whiff_reward)

    # apply reward discounts
    eval_state_df = apply_reward_discount(eval_state_df, reward_gamma)

    # remove rewards during hit state
    eval_state_df = remove_rewards_during_hit(eval_state_df)

    # trim full df down to just state, action, reward
    output_with_input_and_reward = trim_reward_df(eval_state_df, 'actual_reward', reaction_delay)

    return output_with_input_and_reward


def apply_reward_discount(df, discount_factor):
    discounted_rewards = []
    cumulative_reward = 0
    rewards = df['reward'].to_list()
    for reward in rewards[::-1]:
        cumulative_reward = reward + cumulative_reward * discount_factor
        discounted_rewards.append(cumulative_reward)
    discounted_rewards = discounted_rewards[::-1]
    df['discounted_reward'] = discounted_rewards
    # df['actual_reward'] = df.apply(lambda x: x['reward'] if abs(x['reward']) > abs(x['discounted_reward']) else x['discounted_reward'], axis=1)
    df['actual_reward'] = df['discounted_reward']

    return df


def remove_rewards_during_hit(df: pd.DataFrame):
    """
        removes reward data when player is being hit as meaningful actions can be preformed during this time
        if enemy player is not hit during the player atk frames
        :param df: the game state, each row is a frame
        """
    p_idx = 0
    hit_col = 'p_{}_hit'.format(p_idx)

    df = df[df[hit_col] == 0]
    # df = df[df['actual_reward'] != 0]

    return df


def apply_motion_type_reward(df: pd.DataFrame, atk_preframes: int, whiff_reward: float):
    """
    for each motion segment if the motion contains a atk apply reward if the attack hits
    or apply whiff reward if the attack misses
    :param p_idx: player to apply whiff reward to
    :param df: the game state, each row is a frame
    :param atk_preframes:
    :param whiff_reward:
    :return: the state dataframe with reward calculated for each frame/row
    """

    p_idx = 0
    motion_type_col = 'p_{}_motion_type'.format(p_idx)
    atk_col = 'p_{}_atk'.format(p_idx)
    enemy_hit_col = 'p_{}_hit'.format(1 - p_idx)

    # calculate atk changes
    v = (df[motion_type_col] != df[motion_type_col].shift()).cumsum()
    u = df.groupby(v)[motion_type_col].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)

    # create motion_type segments
    motion_type_segment = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]

    # calculate hit changes
    v = (df[enemy_hit_col] != df[enemy_hit_col].shift()).cumsum()
    u = df.groupby(v)[enemy_hit_col].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)

    hit_motion_segments = []
    # create motion_type segments
    hit_change_segments = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
    for hs in hit_change_segments:
        # find the start of the motion type
        hit_motion_type = df.loc[hs[0], 'p_0_motion_type']
        hit_motion_start = hs[0]
        while hit_motion_type == df.loc[hit_motion_start - 1, 'p_0_motion_type']:
            hit_motion_start = hit_motion_start - 1
        hit_motion_segments.append(hit_motion_start)

    # apply reward for each motion seg
    for hs in motion_type_segment:
        if (df[hs[0]:hs[1]][atk_col].sum() > 1) or (hs[0] in hit_motion_segments):  # if motion has attack in it or hits
            if df[hs[0]:hs[1]][enemy_hit_col].sum() > 1:  # if motion hits
                reward_value = df[(df.index >= hs[0]) & (df.index <= hs[1])]['p_1_health_diff'].sum()
            else:
                reward_value = whiff_reward
            df.loc[(df.index >= hs[0] - atk_preframes) & (df.index < hs[0]), 'reward'] = \
                df.loc[
                    (df.index >= hs[0] - atk_preframes) & (df.index < hs[0]), 'reward'] + reward_value  # apply reward

    return df


def apply_negative_motion_type_reward(df: pd.DataFrame, atk_preframes: int, whiff_reward: float):
    """
    for each motion segment if the motion contains a atk apply reward if the attack hits
    or apply whiff reward if the attack misses
    :param p_idx: player to apply whiff reward to
    :param df: the game state, each row is a frame
    :param atk_preframes:
    :param whiff_reward:
    :return: the state dataframe with reward calculated for each frame/row
    """

    p_idx = 0
    motion_type_col = 'p_{}_motion_type'.format(p_idx)
    atk_col = 'p_1_atk'
    player_hit_col = 'p_0_hit'

    df[motion_type_col] = df[motion_type_col].replace(0, -1)

    # calculate atk changes
    v = (df[motion_type_col] != df[motion_type_col].shift()).cumsum()
    u = df.groupby(v)[motion_type_col].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)

    # create motion_type segments
    motion_type_segment = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]

    # calculate hit changes
    v = (df[player_hit_col] != df[player_hit_col].shift()).cumsum()
    u = df.groupby(v)[player_hit_col].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)

    hit_motion_segments_value = {}
    # create motion_type segments
    hit_change_segments = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]
    for hs in hit_change_segments:
        reward_value = df[(df.index >= hs[0]) & (df.index <= hs[1])][
            'p_0_health_diff'].sum()  # sum all damage for the entire hit segment

        # find the start of the motion type
        hit_motion_type = df.loc[hs[0] - 1, 'p_0_motion_type']
        hit_motion_start = hs[0] - 1
        found_prev_motion_type = False
        while not found_prev_motion_type:
            hit_motion_start = hit_motion_start - 1
            if hit_motion_start - 1 == -1:
                break
            if hit_motion_type != df.loc[hit_motion_start - 1, 'p_0_motion_type'] and df.loc[
                hit_motion_start, 'p_0_hit'] == 0:
                found_prev_motion_type = True
        if found_prev_motion_type:
            hit_motion_segments_value[hit_motion_start] = reward_value

    # apply reward for each motion seg
    for hs in motion_type_segment:
        if hs[0] in hit_motion_segments_value:  # if motion has attack in it or hits
            df.loc[(df.index >= hs[0] - atk_preframes) & (df.index < hs[0]), 'reward'] = \
                df.loc[(df.index >= hs[0] - atk_preframes) & (df.index < hs[0]), 'reward'] + hit_motion_segments_value[
                    hs[0]]  # apply reward

    return df


def generate_json_from_in_out_df(output_with_input_and_reward: pd.DataFrame):
    """
    creates reward json from reward dataframe
    :param output_with_input_and_reward:
    :return:
    """
    json_dict = {}

    state_columns = [c for c in output_with_input_and_reward.columns if c.startswith('state_')]
    action_columns = [c for c in output_with_input_and_reward.columns if c.startswith('input')]
    json_dict['state'] = output_with_input_and_reward[state_columns].values.tolist()
    json_dict['reward'] = output_with_input_and_reward['reward'].values.tolist()
    json_dict['action'] = output_with_input_and_reward[action_columns].values.tolist()

    return json_dict


def generate_rewards(eval_path: str, reward_path: str, reward_columns: dict, falloff: int, player_idx: int,
                     reaction_delay: int, atk_preframes: int, whiff_reward: float,
                     reward_gamma: int) -> str:
    """
    loads all the eval files contained in the eval path and generates a reward file for each
    :param eval_path:
    :param reward_path:
    :param reward_columns: state column name and scaling used to calculate reward
    :param falloff: old gamma - not used
    :param player_idx:
    :param reaction_delay:
    :param atk_preframes:
    :param whiff_reward:
    :param reward_gamma:
    :return: path to new reward file
    """

    Path("{}".format(reward_path)).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(eval_path) if isfile(join(eval_path, f))]

    for file in onlyfiles:
        try:
            if file.startswith('eval_'):
                reward_file = file.replace('eval_', 'reward_')
                file_name = "{}/{}".format(eval_path, file)
                df = calculate_reward_from_eval(
                    file_path=file_name,
                    reward_columns=reward_columns,
                    falloff=falloff, player_idx=player_idx,
                    reaction_delay=reaction_delay,
                    atk_preframes=atk_preframes,
                    whiff_reward=whiff_reward,
                    reward_gamma=reward_gamma)
                file_json = generate_json_from_in_out_df(df)

                with open("{}/{}".format(reward_path, reward_file), 'w') as f_writer:
                    f_writer.write(json.dumps(file_json))

        except Exception as e:
            logger.debug("file_name={}/{}".format(eval_path, file))
            logger.error(e)
            raise e

    return reward_path


def main():
    pass
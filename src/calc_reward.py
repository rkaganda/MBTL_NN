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


def trim_reward_df(eval_state_df: pd.DataFrame, reward_column: str) -> pd.DataFrame:
    """
    remove all columns not needed to generate reward file
    :param eval_state_df:
    :param reward_column:
    :return:
    """
    eval_state_df.rename(columns={reward_column: "reward"}, inplace=True)
    state_columns = [c for c in eval_state_df.columns if c.startswith('state')]
    eval_state_df = eval_state_df[state_columns + ['reward'] + ['input']]

    eval_state_df = eval_state_df[:-1]

    return eval_state_df


def calculate_reward_from_eval(
        file_path: str,
        reward_columns: dict,
        falloff: int,
        player_idx: int,
        reaction_delay: int,
        hit_preframes: int,
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
    :param hit_preframes: how many input frames to before the hit to apply the reward to
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

    # apply rewards for hitting and getting hit
    eval_state_df = apply_hit_segment_rewards(eval_state_df, hit_preframes)

    # apply whiff rewards
    eval_state_df = apply_whiff_reward(eval_state_df, atk_preframes, whiff_reward)

    # apply reward discounts
    eval_state_df = apply_reward_discount(eval_state_df, reward_gamma)

    # trim full df down to just state, action, reward
    output_with_input_and_reward = trim_reward_df(eval_state_df, 'reward_total_norm')

    return output_with_input_and_reward


def apply_reward_discount(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """
    apply custom discount reward function to
    :param df:
    :param gamma:
    :return:
    """
    df['discounted_reward'] = df['reward']
    for n in range(0, 5):
        df['discounted_reward'] = (df['discounted_reward'] + df['discounted_reward'].shift(-1) * gamma) * .5

        df['actual_reward'] = df.apply(
            lambda x: x['reward'] if abs(x['reward']) > abs(x['discounted_reward']) else x['discounted_reward'], axis=1)

    return df


def apply_hit_segment_rewards(df: pd.DataFrame, hit_preframes: int) -> pd.DataFrame:
    """
    generates reward for each frame starting from n (hit_preframes) before the enemy was hit to the last hitframe
    the amount of reward applied each frame is the sum differences in health each frame from the start to end of hit stun
    if the player is hit the reward is negative, if hitting reward is positive
    :param df: the game state, each row is a frame
    :param hit_preframes:
    :return: the state dataframe with reward calculated for each frame/row
    """
    # get start and stop of the hit
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

    return df


def apply_whiff_reward(df: pd.DataFrame, atk_preframes: int, whiff_reward: float):
    """
    generates negative reward for each frame starting from player atk - atk_preframes to player atk start
    if enemy player is not hit during the player atk frames
    :param p_idx: player to apply whiff reward to
    :param df: the game state, each row is a frame
    :param atk_preframes:
    :param whiff_reward:
    :return: the state dataframe with reward calculated for each frame/row
    """

    p_idx = 0
    atk_col = 'p_{}_atk'.format(p_idx)
    enemy_hit_col = 'p_{}_hit'.format(1 - p_idx)

    # calculate atk changes
    v = (df[atk_col] != df[atk_col].shift()).cumsum()
    u = df.groupby(v)[atk_col].agg(['all', 'count'])
    m = u['all'] & u['count'].ge(1)

    # create atk segments
    atk_segements = df.groupby(v).apply(lambda x: (x.index[0], x.index[-1]))[m]

    # find whiffs
    for hs in atk_segements:
        if df[hs[0]:hs[1]][enemy_hit_col].sum() < 1:  # if atk whiffed
            df.loc[(df.index <= hs[0]) & (df.index > hs[0] - atk_preframes), 'reward'] = \
                df.loc[(df.index <= hs[0]) & (
                            df.index > hs[0] - atk_preframes), 'reward'] + whiff_reward  # apply whiff reward

    return df


def apply_invalid_input_reward(e_df: pd.DataFrame, reaction_delay: int,
                               neutral_action_index: int) -> pd.DataFrame:
    """
    reward function that applied a negative reward if action had no effect on state (no motion type change)
    test results - negative pred q resulted in no actions even if positive reward was possible with different action
    :param e_df:
    :param reaction_delay:
    :param neutral_action_index:
    :return:
    """
    player_idx = 0
    invalid_frame_window = 5
    motion_change_col = 'p_{}_motion_change'.format(player_idx)
    motion_type_col = 'p_{}_motion_type'.format(player_idx)

    e_df[motion_change_col] = e_df[motion_type_col].diff()
    # change_index = e_df[e_df[motion_change_col] != 0].index.tolist()
    change_index = e_df[e_df[motion_change_col] != 0].index.tolist()

    valid_input_index = [v - 1 for v in change_index]
    neutral_inputs = e_df[e_df['input'] == neutral_action_index].index.tolist()

    valid_windows = set(
        itertools.chain.from_iterable([list(range(v - invalid_frame_window, v)) for v in valid_input_index])).union(
        neutral_inputs)
    valid_windows = [v for v in valid_windows if v > 0]
    invalid_index = e_df[~e_df.index.isin(valid_windows)].index.tolist()
    invalid_index = [v for v in invalid_index if v > reaction_delay]

    e_df.loc[e_df.index[invalid_index], 'reward_total_norm'] = -1000

    return e_df


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
                     reaction_delay: int, hit_preframes: int, atk_preframes: int, whiff_reward: float,
                     reward_gamma: int) -> str:
    """
    loads all the eval files contained in the eval path and generates a reward file for each
    :param eval_path:
    :param reward_path:
    :param reward_columns: state column name and scaling used to calculate reward
    :param falloff: old gamma - not used
    :param player_idx:
    :param reaction_delay:
    :param hit_preframes:
    :param atk_preframes:
    :param whiff_reward:
    :param reward_gamma:
    :return: path to new reward file
    """
    print(reward_path)
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
                    hit_preframes=hit_preframes,
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

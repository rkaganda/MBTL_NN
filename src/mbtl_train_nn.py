from ctypes import windll
import time
import datetime
import json
from pathlib import Path

from mem_access_util import mem_util
import mem_access_util

import cfg_tl
import sub_tl

import multiprocessing as mp
from multiprocessing import Process, Manager, Event, Value


import logging

import config
import mbtl_input
import melty_state
from nn.eval_worker import EvalWorker
import nn.rnn_model
import nn.model_util

state_format = dict()
state_format['directions'] = config.settings['directions']
state_format['buttons'] = config.settings['buttons']
state_format['minmax'] = melty_state.get_minmax()
state_format['attrib'], attrib_keys = melty_state.get_attributes()
state_format['action_max'] = {}
state_format['values'] = melty_state.get_values()
state_format['categories'], state_format['categorical'] = melty_state.get_categories()


logging.basicConfig(filename='../logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def get_state_data(cfg: cfg_tl) -> dict:
    character_elements = cfg.characters_data_list[0]
    state_data = dict()
    for character_idx in [0, 1]:
        state_data[character_idx] = dict()
        for n_ in attrib_keys:  # for each attribute in character data
            cd_attrib = getattr(character_elements.characters_data[character_idx], n_)  # get the attribute
            # if the attribute is memory data
            if isinstance(cd_attrib, mem_access_util.mem_util.Mem_Data_Class):
                state_data[character_idx][n_] = cd_attrib.r_mem()

    return state_data


def monitor_state(
        game_states,
        timer_log,
        env_status,
        eval_statuses,
        timer_max,
        action_buffer,
        current_state_frame,
        current_action
):
    cfg = cfg_tl
    sub = sub_tl

    windll.winmm.timeBeginPeriod(1)  # タイマー精度を1msec単位にする # Set timer accuracy to 1msec unit

    # ベースアドレス取得 # Get base address
    mem_util.get_connection("MBTL.exe")

    # check_data格納用オブジェクト作成
    cfg.game_data = cfg_tl.Game_Data_Class()
    for n in range(3):  # 3フレーム分のデータ格納領域作成
        cfg.characters_data_list.append(cfg_tl.Characters_Data_Class())

    timer_old = -1
    timer = cfg.game_data.timer.r_mem()
    data_index = 0
    round_reset = False

    while not env_status['die']:
        time.sleep(0.001)
        # トレーニングモードチェック
        tr_flag = cfg.game_data.tr_flag.r_mem()

        if tr_flag == 100 or tr_flag == 300:  # if in verses mode or training mode
            sub.function_key(data_index)
            time.sleep(.001)
            while all([_v['eval_ready'] for _, _v in eval_statuses.items()]):
                if not round_reset:
                    reset_timer = cfg.game_data.timer.r_mem()
                    while cfg.game_data.timer.r_mem() > reset_timer:
                        round_reset = True
                        mbtl_input.reset_round()
                        time.sleep(.001)  # wait for round to reset
                    timer_old = cfg.game_data.timer.r_mem()
                else:
                    pass
                # タイマーチェック # timer check
                timer = cfg.game_data.timer.r_mem()

                # フレームの切り替わりを監視 # Monitor frame switching
                if timer != timer_old and round_reset:  # if new frame
                    if (timer < timer_old) or timer > timer_max:  # if timer is reset (new match)
                        env_status['round_done'] = True
                        for _, eval_status in eval_statuses.items():
                            eval_status['eval_ready'] = False
                        round_reset = False
                        print("round done.. stopping eval")
                        break
                    timer_log.append(timer)
                    game_states.append({
                        'game': get_state_data(cfg),
                        'input': [v.value for _, v in current_action.items()]
                    })
                    with current_state_frame.get_lock():
                        current_state_frame.value = len(game_states)

                    timer_old = timer  # store last time data was logged
                    time.sleep(0.004)  # データが安定するまで待機 # Wait for data to stabilize

                    sub.situationCheck(data_index)  # 各種数値の取得 # Get various values

                    sub.content_creation(data_index)  # 各種データ作成 # Various data creation
                else:
                    pass
    logger.debug("monitor dead")


def capture_rounds(round_num: int):
    # create data structures
    manager = Manager()
    game_states_ = manager.list()  # stores game state for each frame
    timer_log_ = manager.list()  # list of frame numbers
    env_status_ = manager.dict()  # stores env status
    env_status_['die'] = False  # kill processes event
    env_status_['round_done'] = False  # round done event

    # params
    timer_max = config.settings['timer_max']
    current_state_frame = Value('i', 0)  # current state frame

    eval_statuses_ = dict()
    eval_workers = dict()
    kill_inputs_process = Event()
    do_inputs_processes = dict()
    player_facing_flags = dict()
    action_buffer = dict()
    current_actions = dict()

    # for each player
    for p in range(0, config.settings['player_count']):
        eval_statuses_[p] = manager.dict()  # share data across processes
        eval_statuses_[p]['kill_eval'] = False  # eval die
        eval_statuses_[p]['storing_eval'] = False  # eval storing data
        eval_statuses_[p]['eval_ready'] = False  # eval ready generate inputs

        action_buffer[p] = manager.dict()
        action_list, neutral_action_index, facing_flag = mbtl_input.create_action_list(p)  # action => key mapping

        # create a list of pre frames
        pre_frame_keys = list(range(config.settings['p{}_model'.format(p)]['reaction_delay']+1))
        # assign the first pre frames to neutral
        for k in pre_frame_keys:
            action_buffer[p][k] = neutral_action_index
        player_facing_flags[p] = manager.dict()  # player facing at frame
        for k in pre_frame_keys:
            player_facing_flags[p][k] = p
        state_format['action_max'][p] = len(action_list) - 1
        current_actions[p] = Value('i', neutral_action_index)

        model_config = nn.model_util.load_model_config(p)

        # create worker for evaluation/training/reward
        eval_w = EvalWorker(
            game_states=game_states_,
            env_status=env_status_,
            eval_status=eval_statuses_[p],
            frames_per_evaluation=int(model_config['frames_per_observation']),
            reaction_delay=int(model_config['reaction_delay']),
            input_index_max=len(action_list[p])-1,
            state_format=state_format,
            learning_rate=float(model_config['learning_rate']),
            player_idx=p,
            frame_list=timer_log_,
            neutral_action_index=neutral_action_index,
            current_state_frame=current_state_frame,
            player_facing_flags=player_facing_flags[p],
            action_buffer=action_buffer[p]
        )
        eval_workers[p] = eval_w

        # process to update actions=>keys each frame
        do_input_process = Process(
            target=mbtl_input.do_inputs,
            args=(action_buffer[p],
                  action_list,
                  kill_inputs_process,
                  env_status_,
                  player_facing_flags[p],
                  current_state_frame,
                  current_actions[p]
                  ))
        do_inputs_processes[p] = do_input_process

    # monitor env and update env state every frame
    monitor_mbtl_process = Process(target=monitor_state,
                                   args=(game_states_,
                                         timer_log_,
                                         env_status_,
                                         eval_statuses_,
                                         timer_max,
                                         action_buffer,
                                         current_state_frame,
                                         current_actions
                                         ))

    print("starting")
    logger.debug("starting")
    for _, eval_worker in eval_workers.items():  # start the eval workers
        eval_worker.start()
    logger.debug("eval started")
    monitor_mbtl_process.start()  # start process to monitor game state
    logger.debug("monitored started")
    for _, inputs_process in do_inputs_processes.items():  # start process to do inputs
        inputs_process.start()
    logger.debug("do inputs started")

    for r in range(0, round_num):  # for each round
        print("round={}".format(r))
        logger.debug("round={}".format(r))
        while not env_status_['round_done']:  # while round is running
            time.sleep(.001)
        logger.debug("round done, notifying eval")
        for _, eval_st in eval_statuses_.items():  # for each player eval
            eval_st['storing_eval'] = True  # eval calc reward, store data, train model
        # wait for eval to finish
        while all([not _v['eval_ready'] for _, _v in eval_statuses_.items()]):
            time.sleep(.001)
        logger.debug("storing eval done")
        logger.debug("clearing buffers")
        del game_states_[:]  # clear stores
        del timer_log_[:]  # clear stores
        logger.debug("resetting round round")
        env_status_['round_done'] = False

    logger.debug("killing evals")
    for _, eval_st in eval_statuses_.items():
        eval_st['kill_eval'] = True
    logger.debug("killing env")
    env_status_['die'] = True
    logger.debug("killing inputs processes")
    kill_inputs_process.set()
    logger.debug("sent")

    logger.debug("eval worker join")
    for _, eval_w in eval_workers.items():
        eval_w.join()
    logger.debug("monitor_mbtl_process join")
    monitor_mbtl_process.join()
    logger.debug("do_p1_inputs_process join")
    for _, eval_w in eval_workers.items():
        eval_w.join()
    for _, inputs_process in do_inputs_processes.items():
        inputs_process.join()

    print("stopped")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    capture_rounds(1000)

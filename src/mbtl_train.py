import copy
import ctypes.wintypes
from ctypes import windll
import os
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
# import torch.multiprocessing as mp
# from torch.multiprocessing import Process, Manager, Event
import logging

import config
import mbtl_input
from nn.EvalWorker import EvalWorker

# load minmax
with open(config.settings['minmax_file']) as f:
    state_format = json.load(f)
    state_format['directions'] = config.settings['directions']
    state_format['buttons'] = config.settings['buttons']

attrib_keys = list(state_format['attrib'])

logging.basicConfig(filename='logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


def get_state_data(cfg):
    character_elements = cfg.characters_data_list[0]
    state_data = dict()
    for character_idx in [0, 1]:
        state_data[character_idx] = dict()
        # for n_ in dir(character_elements.characters_data[character_idx]):  # for each attribute in character data
        for n_ in attrib_keys:  # for each attribute in character data
            # if n_ != "bunker_pointer" and n_ != "c_timer":
            cd_attrib = getattr(character_elements.characters_data[character_idx], n_)  # get the attribute
            # if the attribute is memory data
            if isinstance(cd_attrib, mem_access_util.mem_util.Mem_Data_Class):
                state_data[character_idx][n_] = cd_attrib.r_mem()

    return state_data


def monitor_state(game_states, input_indices, timer_log, env_status, eval_statuses, timer_max):
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
                        'input': [v.value for _, v in input_indices.items()]
                    })

                    timer_old = timer  # store last time data was logged
                    time.sleep(0.004)  # データが安定するまで待機 # Wait for data to stabilize

                    sub.situationCheck(data_index)  # 各種数値の取得 # Get various values

                    sub.content_creation(data_index)  # 各種データ作成 # Various data creation
                else:
                    pass
    logger.debug("monitor dead")


def store_states(states, timer_log):
    full_store = {}
    for idx, timestamp in enumerate(timer_log):
        full_store[str(timestamp)] = {
            "state": states[str(idx)]['game'],
            "inputs": states[str(idx)]['input']
        }
    match_dir = "data/rounds"  # dir to store matches
    Path("{}/{}".format(match_dir, config.settings['run_name'])).mkdir(parents=True, exist_ok=True)

    with open("{}/{}.json".format(match_dir, int(datetime.datetime.now().timestamp())), 'a') as f_writer:
        f_writer.write(json.dumps(full_store))


def capture_rounds(round_num):
    # create data structures
    manager = Manager()
    game_states_ = manager.list()
    timer_log_ = manager.list()
    env_status_ = manager.dict()
    env_status_['die'] = False
    env_status_['round_done'] = False

    # params
    frames_per_observation = int(config.settings['frames_per_observation'])
    reaction_delay = int(config.settings['reaction_delay'])
    learning_rate = float(config.settings['learning_rate'])
    timer_max = config.settings['timer_max']

    eval_statuses_ = dict()
    eval_workers = dict()
    kill_inputs_process = Event()
    do_inputs_processes = dict()
    input_indices = dict()

    for p in range(0, 1):
        eval_statuses_[p] = manager.dict()
        eval_statuses_[p]['kill_eval'] = False  # eval die
        eval_statuses_[p]['storing_eval'] = False  # eval storing data
        eval_statuses_[p]['eval_ready'] = False  # eval ready generate inputs

        input_list, neutral_index = mbtl_input.create_input_list(0)
        input_indices[p] = Value('i', neutral_index)

        eval_w = EvalWorker(
            game_states=game_states_,
            env_status=env_status_,
            eval_status=eval_statuses_[p],
            frames_per_evaluation=frames_per_observation,
            reaction_delay=reaction_delay,
            input_index=input_indices[p],
            input_index_max=len(input_list)-1,
            state_format=state_format,
            learning_rate=learning_rate,
            player_idx=p,
            frame_list=timer_log_,
        )
        eval_workers[p] = eval_w

        do_input_process = Process(
            target=mbtl_input.do_inputs,
            args=(input_indices[p], input_list, kill_inputs_process, env_status_))
        do_inputs_processes[p] = do_input_process

    monitor_mbtl_process = Process(target=monitor_state,
                                   args=(game_states_, input_indices, timer_log_, env_status_, eval_statuses_, timer_max))

    print("starting")
    logger.debug("starting")
    for _, eval_worker in eval_workers.items():
        eval_worker.start()
    logger.debug("eval started")
    monitor_mbtl_process.start()
    logger.debug("monitored started")
    for _, inputs_process in do_inputs_processes.items():
        inputs_process.start()
    logger.debug("do inputs started")

    for r in range(0, round_num):
        print("round={}".format(r))
        logger.debug("round={}".format(r))
        while not env_status_['round_done']:
            time.sleep(.001)
        logger.debug("round done, notifying eval")
        for _, eval_st in eval_statuses_.items():
            eval_st['storing_eval'] = True
        # wait for eval to finish storing
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
    # status_['kill_eval'] = True
    logger.debug("killing env")
    env_status_['die'] = True
    logger.debug("killing inputs processes")
    kill_inputs_process.set()
    logger.debug("sent")

    logger.debug("eval worker join")
    for _, eval_w in eval_workers.items():
        eval_w.join()
    # module_eval_worker.join()
    logger.debug("monitor_mbtl_process join")
    monitor_mbtl_process.join()
    logger.debug("do_p1_inputs_process join")
    for _, eval_w in eval_workers.items():
        eval_w.join()
    for _, inputs_process in do_inputs_processes.items():
        inputs_process.join()

    print("stopped")


def collect_data(capture_count):
    capture_rounds(capture_count)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    collect_data(1)

    # test_no_inputs()

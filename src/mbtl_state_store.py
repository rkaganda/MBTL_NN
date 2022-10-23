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
from multiprocessing import Process, Manager, Event
# import torch.multiprocessing as mp
# from torch.multiprocessing import Process, Manager, Event

import config
import mbtl_input
from nn.EvalWorker import EvalWorker

# load minmax
with open(config.settings['minmax_file']) as f:
    state_format = json.load(f)
    state_format['input'] = config.settings['valid_inputs']

attrib_keys = list(state_format['attrib'])


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


def monitor_state(game_states, input_state, timer_log, status, timer_max):
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
    round_started = False

    while not status['die']:
        time.sleep(0.001)
        # トレーニングモードチェック
        tr_flag = cfg.game_data.tr_flag.r_mem()

        if tr_flag == 100 or tr_flag == 300:  # if in verses mode or training mode
            sub.function_key(data_index)

            if not round_started:
                mbtl_input.reset_round()
                round_started = True
                while cfg.game_data.timer.r_mem() > timer:
                    time.sleep(.013)  # wait for round to reset
            # タイマーチェック # timer check
            timer = cfg.game_data.timer.r_mem()

            # フレームの切り替わりを監視 # Monitor frame switching
            if timer != timer_old and round_started:  # if new frame
                if (timer < timer_old) or timer > timer_max:  # if timer is reset (new match)
                    status['round_finished'] = True
                    break

                timer_log.append(timer)
                game_states.append({
                    'game': get_state_data(cfg),
                    'input': copy.deepcopy(input_state)
                })
                timer_old = timer  # store last time data was logged
                time.sleep(0.004)  # データが安定するまで待機 # Wait for data to stabilize

                sub.situationCheck(data_index)  # 各種数値の取得 # Get various values

                sub.content_creation(data_index)  # 各種データ作成 # Various data creation
    print("monitor dead")


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


def capture_round():
    # create data structures
    game_states_ = Manager().list()
    timer_log_ = Manager().list()
    status_ = Manager().dict()
    status_['die'] = False
    status_['round_finished'] = False

    p1_input_dict = mbtl_input.create_p1_input_dict()

    # generate_inputs = Event()
    do_inputs = Event()
    kill_eval = Event()
    eval_ready = Event()

    # params
    frames_per_observation = 5
    reaction_delay = 5
    learning_rate = 1e-5
    timer_max = 2000

    # create workers and processes
    # generate_p1_inputs_process = Process(
    #     target=mbtl_input.random_inputs,
    #     args=(p1_input_dict, generate_inputs))
    do_p1_inputs_process = Process(
        target=mbtl_input.do_inputs,
        args=(p1_input_dict, mbtl_input.p1_mapping_dict, do_inputs))

    # p2_input_dict = mbtl_input.create_p2_input_dict()
    # generate_p2_inputs_process = Process(
    #     target=mbtl_input.random_inputs,
    #     args=(p2_input_dict, generate_inputs))
    # do_p2_inputs_process = Process(
    #     target=mbtl_input.do_inputs,
    #     args=(p2_input_dict, mbtl_input.p2_mapping_dict, do_inputs))

    monitor_mbtl_process = Process(target=monitor_state,
                                   args=(game_states_, p1_input_dict, timer_log_, status_, timer_max))

    module_eval_worker = EvalWorker(
        game_states=game_states_,
        die=kill_eval,
        frames_per_evaluation=frames_per_observation,
        reaction_delay=reaction_delay,
        input_state=p1_input_dict,
        state_format=state_format,
        learning_rate=learning_rate,
        player_idx=0,
        frame_list=timer_log_,
        worker_ready=eval_ready
    )

    print("starting")
    module_eval_worker.start()
    while not eval_ready.is_set():
        time.sleep(.001)
    print("eval ready")
    monitor_mbtl_process.start()
    print("monitored started")
    do_p1_inputs_process.start()
    print("do inputs started")

    # generate_p2_inputs_process.start()
    # do_p2_inputs_process.start()

    while not status_['round_finished']:
        time.sleep(.13)

    print("sending die")
    status_['die'] = True
    kill_eval.set()
    do_inputs.set()
    print("sent")

    print("eval worker join")
    module_eval_worker.join()
    print("monitor_mbtl_process join")
    monitor_mbtl_process.join()
    print("do_p1_inputs_process join")
    do_p1_inputs_process.join()

    # generate_inputs.set()
    # generate_p1_inputs_process.join()
    # generate_p2_inputs_process.join()
    # print("generate_p2_inputs_process join")
    # do_p2_inputs_process.join()
    # print("do_p2_inputs_process join")

    print("stopped")

    # store_states(game_states_, timer_log_)


def collect_data(capture_count):
    for c in range(0, capture_count):
        print("round={}".format(c))
        capture_round()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    collect_data(20)

    # test_no_inputs()

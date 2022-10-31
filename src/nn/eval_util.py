import copy
import json
import datetime
from functools import singledispatch
from pathlib import Path
import config
import torch


def store_eval_output(normalized_states, states, model_output, state_format, player_idx, episode_number):
    path = "data/eval/{}".format(config.settings['run_name'])
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    states = copy.deepcopy(states)
    for idx, v in enumerate(states):
        states[idx]['input'] = states[idx]['input'][player_idx]

    all_data = {
        "normalized_states": normalized_states,
        "states": copy.deepcopy(states),
        "model_output": copy.deepcopy(model_output),
        "state_format": state_format
    }
    Path("{}/evals/{}/{}".format(path, player_idx, episode_number)).mkdir(parents=True, exist_ok=True)
    Path("{}/model/{}/{}".format(path, player_idx, episode_number)).mkdir(parents=True, exist_ok=True)
    Path("{}/stats/{}".format(path, player_idx)).mkdir(parents=True, exist_ok=True)

    with open("{}/evals/{}/{}/eval_{}.json".format(path, player_idx, episode_number, datetime_str),'a') as f_writer:
        f_writer.write(json.dumps(all_data))

    with open("{}/config.json".format(path),'w') as f_writer:
        f_writer.write(json.dumps(config.settings))






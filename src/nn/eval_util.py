import copy
import json
import datetime
from functools import singledispatch
from pathlib import Path
import config
import torch

@singledispatch
def keys_to_strings(ob):
    return ob


@keys_to_strings.register
def _handle_dict(ob: dict):
    return {str(k): keys_to_strings(v) for k, v in ob.items()}


@keys_to_strings.register
def _handle_list(ob: list):
    return [keys_to_strings(v) for v in ob]


def store_eval_output(normalized_states, states, model_output, state_format):
    path = "data/eval/{}".format(config.settings['run_name'])

    all_data = {
        "normalized_states": normalized_states,
        "states": copy.deepcopy(states),
        "model_output": copy.deepcopy(model_output),
        "state_format": state_format
    }
    Path("{}".format(path)).mkdir(parents=True, exist_ok=True)
    Path("{}/model/".format(path)).mkdir(parents=True, exist_ok=True)

    with open("{}/eval_{}.json".format(path, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), 'a') as f_writer:
        f_writer.write(json.dumps(all_data))






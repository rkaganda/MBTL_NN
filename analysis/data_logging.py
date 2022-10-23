from src import cfg
import mem_access_util
import ctypes.wintypes
import difflib
import json
import datetime
import sys
import os


def display_elements(elements, old_data):
    data_dict = dict()
    changed_data = None
    # [Characters_Data_Class]
    for idx, e_ in enumerate(elements):
        if idx in [0]:
            # print(e_.characters_data)
            # [Character_Data_Class]
            for jdx, cd_ in enumerate(e_.characters_data):
                if jdx in [0, 1]:  # if player 1 or 2....
                    if jdx not in data_dict:  # if no dict for player data
                        data_dict[jdx] = dict()  # create dict for current data
                    # print(dir(cd_))
                    for n_ in dir(cd_):  # for each attribute in character ata
                        if str(n_) != "bunker_pointer" and str(n_) != "c_timer":
                            cd_attrib = getattr(cd_, n_)  # get the attribute
                            # if the attribute is memory data
                            if isinstance(cd_attrib, mem_access_util.mem_util.Mem_Data_Class):
                                if n_ not in data_dict[jdx]:  # if there is no key for this attribute
                                    data_dict[jdx][n_] = dict()  # create a key to store this attribute
                                data_dict[jdx][n_] = cd_attrib.r_mem()  # read the attribute data from memory
                                if old_data is not None:  # if there is old data
                                    if old_data[jdx][n_] != data_dict[jdx][n_]:  # if the attribute has changed
                                        changed_data = update_changed_data(
                                            player_index=jdx,
                                            attribute_name=n_,
                                            value=data_dict[jdx][n_],
                                            changed_data=changed_data
                                        )
                                        changed_data[jdx][n_] = data_dict[jdx][n_]  # store the change
                                else:  # if there is no old data
                                    changed_data = update_changed_data(jdx, n_, data_dict[jdx][n_], changed_data)
    return data_dict, changed_data


def update_changed_data(player_index, attribute_name, value, changed_data):
    if changed_data is None:
        changed_data = dict()
    if player_index not in changed_data:
        changed_data[player_index] = dict()
    if attribute_name not in changed_data[player_index]:
        changed_data[player_index][attribute_name] = value

    return changed_data


def store_dict_as_csv(match_data_queue):
    match_dir = "rounds"  # dir to store matches
    if not os.path.exists(match_dir):
        os.mkdir("rounds")
    while True:
        if not match_data_queue.empty():
            data = match_data_queue.get()
            if data == -1:
                print("closing queue")
                break
            else:
                print("writing match_data...", end="")
                with open("{}/{}.json".format(match_dir, int(datetime.datetime.now().timestamp())), 'a') as f_writer:
                    f_writer.write(json.dumps(data))
                print("done.")




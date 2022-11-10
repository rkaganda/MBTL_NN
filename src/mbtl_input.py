import copy
import ctypes
import multiprocessing
import random
from ctypes import wintypes
import time
import logging
from multiprocessing.sharedctypes import Synchronized
from multiprocessing import Process, Manager, Event, Value

import config
from typing import Tuple

logging.basicConfig(filename='../logs/train.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

# msdn.microsoft.com/en-us/library/dd375731

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", wintypes.ULONG_PTR))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))


LPINPUT = ctypes.POINTER(INPUT)


def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT,  # nInputs
                             LPINPUT,  # pInputs
                             ctypes.c_int)  # cbSize


# Functions

def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


# action to game input mapping
mapping_dicts = dict()

mapping_dicts[0] = dict()
mapping_dicts[0]['directions'] = dict()
mapping_dicts[0]['directions']['1'] = [0x53, 0x41]  # P1 1 = 4+2
mapping_dicts[0]['directions']['2'] = [0x53]  # P1 2 # s
mapping_dicts[0]['directions']['3'] = [0x53, 0x57]  # P1 3 = 2+6
mapping_dicts[0]['directions']['4'] = [0x41]  # P1 4 # a
mapping_dicts[0]['directions']['5'] = []  # P1 5
mapping_dicts[0]['directions']['6'] = [0x57]  # P1 6 # w
mapping_dicts[0]['directions']['7'] = [0x57, 0x44]  # P1 7 = 4 + 8
mapping_dicts[0]['directions']['8'] = [0x44]  # P1 8 # d
mapping_dicts[0]['directions']['9'] = [0x44, 0x57]  # P1 8 # d

# buttons
mapping_dicts[0]['buttons'] = dict()
mapping_dicts[0]['buttons']['a'] = [0x49]  # P1 A # i
mapping_dicts[0]['buttons']['b'] = [0x55]  # P1 B # u
mapping_dicts[0]['buttons']['c'] = [0x4F]  # P1 C # o
mapping_dicts[0]['buttons']['d'] = [0x4A]  # P1 D # j

all_keys = [0x53, 0x41, 0x57, 0x44, 0x49, 0x55, 0x4F, 0x4A]

# P2
mapping_dicts[1] = dict()

# directions
mapping_dicts[1]['directions'] = dict()
mapping_dicts[1]['directions']['1'] = [0x58, 0x56]  # P2 2 4
mapping_dicts[1]['directions']['2'] = [0x58]  # P2 2 # x #
mapping_dicts[1]['directions']['3'] = [0x58, 0x43]  # P2 2 6
mapping_dicts[1]['directions']['4'] = [0x56]  # P2 4 # c
mapping_dicts[1]['directions']['5'] = []  # P2 4 # c
mapping_dicts[1]['directions']['6'] = [0x43]  # P2 6 # z
mapping_dicts[1]['directions']['7'] = [0x56, 0x5A]  # P2 4 8
mapping_dicts[1]['directions']['8'] = [0x5A]  # P2 8 # v
mapping_dicts[1]['directions']['9'] = [0x43, 0x5A]  # P2 6 8

# buttons
mapping_dicts[1]['buttons'] = dict()
mapping_dicts[1]['buttons']['a'] = [0x42]  # P2 A # b
mapping_dicts[1]['buttons']['b'] = [0x4E]  # P2 B # n
mapping_dicts[1]['buttons']['c'] = [0x4D]  # P2 C # m
mapping_dicts[1]['buttons']['d'] = [0xBC]  # P2 D # ,


def create_input_list(player_index: int) -> Tuple[list, int]:
    """
    generates list where index each is an action and the item is a list of keycodes for that action
    :param player_index:
    :return: list of keycodes, the index for the neutral/empty action
    """
    directions = []  # stores all possible key combinations
    for idx, direction in enumerate(config.settings['directions']):  # populate list with each key combination
        directions.append(mapping_dicts[player_index]['directions'][direction])

    directions_lists = []  # list for each possible direction
    for r in range(0, len(directions)):
        ar = [0] * 9
        ar[r] = 1
        directions_lists.append(ar)

    for d_list in directions_lists:  # populate direction list with key combinations
        for idx, value in enumerate(directions):
            if d_list[idx] == 1:
                d_list[idx] = value
            else:
                d_list[idx] = []

    # create button combinations list
    buttons_list = []
    for idx, button in enumerate(config.settings['buttons']):
        buttons_list.append(mapping_dicts[player_index]['buttons'][button])

    combinations = [
        [0, 0, 0, 0],  # neutral
        [1, 0, 0, 0],  # a
        [0, 1, 0, 0],  # b
        [0, 0, 1, 0],  # c
        [0, 0, 0, 1],  # d
        [1, 1, 0, 0],  # ab
        [0, 1, 1, 0],  # ab
        [1, 0, 0, 1]  # ac
    ]

    for c_list in combinations:
        for idx, value in enumerate(buttons_list):
            if c_list[idx] == 1:
                c_list[idx] = value
            else:
                c_list[idx] = []

    all_combinations = []

    for d in directions_lists:
        for c in combinations:
            all_combinations.append(d + c)

    combined = []
    neutral_index = None
    for idx, ar in enumerate(all_combinations):  # create list with all direction + button combinations
        new_ar = []
        for val_ar in ar:
            new_ar = new_ar + val_ar
        combined.append(new_ar)
        if len(new_ar) == 0:
            neutral_index = idx

    if neutral_index is None:
        neutral_index = 0

    return combined, neutral_index


def do_inputs(input_index, input_list: list, die, env_status):
    """
    keeps tracks and processes inputs each frame
    :param input_index: the inputs for the current frame
    :param input_list:  list of keys corresponding to each input/action
    :param die: returns when die is set
    :param env_status:
    :return:
    """
    inputs_held = set()  # keys being held
    inputs_hold = set()  # keys to hold
    inputs_cleared = False

    while not die.is_set():
        if not env_status['round_done']:  # if round is live
            time.sleep(.013)  # sleep a frame
            inputs_hold.clear()  # clear held inputs buffer
            for k in input_list[input_index.value]:  # for each key
                if k not in inputs_held:  # if key is not already being held
                    PressKey(k)  # press key
                inputs_hold.add(k)  # add key to hold
            release_inputs = (inputs_held - inputs_hold)  # keys that where pressed last frame but not this frame
            if len(release_inputs) > 0:  # if there are no keys to release
                for k in release_inputs:
                    ReleaseKey(k)
            release_inputs.clear()
            inputs_held = copy.deepcopy(inputs_hold)  # hold inputs till next frame
        else:  # round is over
            if not inputs_cleared:  # have all the keys been released
                for k in inputs_held:  # release all keys being held
                    ReleaseKey(k)
            time.sleep(.001)


def reset_round():
    """
    resets practice mode after the round ends
    :return:
    """
    time.sleep(.001)
    # make sure every key is released
    for k in all_keys:
        time.sleep(.001)
        ReleaseKey(k)

    time.sleep(.001)
    PressKey(0x53)  # s # down
    time.sleep(.001)
    PressKey(0x52)  # r

    time.sleep(.005)
    ReleaseKey(0x52)  # r
    time.sleep(.001)
    ReleaseKey(0x53)  # s # down
    # time.sleep(.001)
